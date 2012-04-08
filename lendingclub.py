import csv
import random
import subprocess
import sys

import numpy
import time
import datetime

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import copy

import fieldparsers
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import sklearn.svm.libsvm
import sklearn.svm


MAX_INTEREST_RATE_TO_INVEST = 1.00
RANDOMIZATION_AMOUNT = 0.002

# Table-based classifier that bins on a single discrete input value and
# averages output values
class BinnedClassifier:

    def __init__(self, csv_col=None):
        self.csv_col = csv_col

    def get_bins(self, A_csv):
        return A_csv[self.csv_col] if self.csv_col != None else numpy.ones(A_csv.shape[0])

    def fit(self, A, A_csv, b):
        bins = self.get_bins(A_csv)
        counts = {}
        sums = {}
        for bin, value in zip(bins, b):
            counts[bin] = counts.setdefault(bin, 0) + 1
            sums[bin] = sums.setdefault(bin, 0) + value

        self.averages = {}
        for k, count in counts.iteritems():
            self.averages[k] = sums[k] / count
        return self

    def predict(self, A, A_csv):
        bins = self.get_bins(A_csv)
        return map(lambda x: self.averages.setdefault(x, 0), A_csv[self.csv_col])

# Classifier that always predicts 1
class TrueValuedClassifier:

    def fit(self, A, A_csv, b):
        return self

    def predict(self, A, A_csv):
        return numpy.ones(A.shape[0])

class SkLearnClassifier:

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def get_classifier_probabilities(self, A):
        probabilities = self.base_classifier.predict_proba(A)
        return probabilities[:,0]

    def predict(self, A, A_csv):
        preds = self.get_classifier_probabilities(A)        
        return preds.T

    def fit(self, A, A_csv, b):
        self.base_classifier.fit(A, b)
        return self

# Normalization / validation / evaluation
class LcLearner:

    def __init__(self, data, csv_data):
        self.data = data.copy()
        self.normalize()
        self.csv_data = csv_data
        self.unnormalized_data = data.copy()

    def normalize(self):
        for c in self.data.dtype.names:
            dc = self.data[c]
            denom = max(dc) - min(dc)
            if denom == 0: denom = 1
            self.data[c] = (dc - min(dc)) / denom

    def construct_matrix(self, cols):
        A = numpy.zeros((len(self.data), len(cols)))
        for i, c in enumerate(cols): A[:,i] = self.data[c]
        return numpy.matrix(A)

    class EvalResults:
        def __init__(self, avg_prediction_error, actual_irates):
            self.avg_prediction_error = avg_prediction_error
            self.actual_irates = numpy.array(actual_irates)
            self.loan_quantities = numpy.array([40, 80, 200, 300, 400, 500, 750, 1000])

        def get_loan_quantities(self):
            return self.loan_quantities

        def return_for_loan_quantities(self):
            return self.actual_irates[self.loan_quantities]

        def __str__(self):
            results = ["Avg pred error: %f" % (self.avg_prediction_error)]
            for x in self.loan_quantities:
                if len(self.actual_irates) >= x:
                    results.append("Return rate top %d investments: %f" % (x, self.actual_irates[x-1]))
            return "\n".join(results)

    # Train model with specified cols and inputs and target_col as output
    def evaluate(self, cols, target_col, classifier):
        split_percent = 0.5
        A = self.construct_matrix(cols)
        b = self.unnormalized_data[target_col]
        split = int(split_percent * len(b))
        A_train = A[1:split,:]
        A_train_csv = self.csv_data[1:split]
        b_train = b[1:split]
        A_test = A[split+1:len(b),:]
        A_test_csv = self.csv_data[split+1:len(b)]
        b_test = b[split+1:len(b)]
        model = classifier.fit(A_train, A_train_csv, b_train)
        preds = model.predict(A_test, A_test_csv)
        errors = preds - b_test
        avg_error = numpy.sum(abs(errors)) / errors.shape[0]

        loan_values = numpy.zeros((len(b_test), ), dtype=[('pred_irate', '>f4'), ('rand', '>f4'), ('actual_irate', '>f4'), ('irate', '>f4')])

        rand_vec = numpy.random.rand(len(b_test))
        loan_values['pred_irate'] = (-1 * A_test_csv['interest_rate'] * preds * 0.01) + (rand_vec * RANDOMIZATION_AMOUNT)
        loan_values['rand'] = rand_vec
        loan_values['actual_irate'] = A_test_csv['interest_rate'] * b_test * 0.01
        loan_values['irate'] = A_test_csv['interest_rate'] * 0.01
        loan_values.sort(order='pred_irate')
        loan_values['pred_irate'] *= -1

        returns = []

        for i, actual_return in enumerate(loan_values['actual_irate']):
            if loan_values['irate'][i] > MAX_INTEREST_RATE_TO_INVEST: continue
            # if i < 100: print "%d: %.4f %.4f %.4f" % (i, actual_return, loan_values['pred_irate'][i], loan_values['irate'][i])
            returns.append(actual_return if len(returns) == 0 else actual_return + returns[-1])

        counts = (1 + numpy.array(range(len(returns))))
        returns = returns / counts

        return self.EvalResults(avg_error, returns)


    def evaluate_all(self, cols, target_col):        
        num_investments = 80
        def create_probabilistic_logistic_classifier():
            clf = sklearn.linear_model.LogisticRegression(C=10000, penalty='l1', scale_C=True)
            return SkLearnClassifier(clf)
        eval_plc = lambda x: self.evaluate(x, target_col, create_probabilistic_logistic_classifier())
        eval_true = lambda: self.evaluate(cols, target_col, TrueValuedClassifier())
        eval_bin = lambda: self.evaluate(cols, target_col, BinnedClassifier(csv_col='credit_grade'))

        num_trials = 20
        return_sums = {'true': 0, 'binned': 0, 'logistic': 0}
        funcs = {'true': eval_true, 'binned': eval_bin, 'logistic': lambda: eval_plc(cols)}
        for i in range(num_trials):
            for k, v in funcs.iteritems():
                return_sums[k] += v().actual_irates[num_investments]
        
        print "Average return rate for %d loans" % (num_investments)
        for k, v in return_sums.iteritems():
            avg = v / num_trials
            print "%20s: %.4f" % (k, avg)
        
        et = eval_true()
        eb = eval_bin()
        ep = eval_plc(cols)

        plt.figure()
        plt.plot(ep.get_loan_quantities(), ep.return_for_loan_quantities(),
                 et.get_loan_quantities(), et.return_for_loan_quantities(),
                 eb.get_loan_quantities(), eb.return_for_loan_quantities())
        plt.ylabel('avg return')
        plt.xlabel('loans invested')
        plt.legend(('logistic regression', 'credit grade binning', 'default rate of 0'))
        plt.savefig("plots/loans_invested.png")

        print "\n\nAssuming no loan defaults:\n%s\n\n" % (et)
        print "Credit grade binning:\n%s\n\n" % (eb)
        print "With all cols:\n%s\n\n" % (ep)

        print "\nReturns for %d investments:" % (num_investments)
        print "%40s %5s %5s" % ("column", "only", "w/o")
        
        print "%40s %.4f %.4f" % ("all", 0.0, eval_plc(cols).actual_irates[num_investments])
        for c in cols:
            cols_copy = copy.copy(cols)
            cols_copy.remove(c)
            print "%40s %.4f %.4f" % (c, eval_plc([c]).actual_irates[num_investments], eval_plc(cols_copy).actual_irates[num_investments])
            #print "%40s %.4f %.4f" % (c, eval_plc([c]).avg_prediction_error, eval_plc(cols_copy).avg_prediction_error)


class LcDataExtractedFeatures:

    def create(self, raw_data):
        self.columns = ['amount_requested', 'interest_rate', 'loan_length', 'application_date', 'credit_grade', 'status', 'one', 'actual_interest_rate', 'debt_to_income_ratio','monthly_income', 'fico_range', 'open_credit_lines', 'total_credit_lines', 'earliest_credit_line_date', 'home_ownership', 'expected_interest_rate', 'loan_id', 'description_length']

        normalizers = {'application_date': self.parse_date,
                       'earliest_credit_line_date': self.parse_date,
                       'credit_grade': self.parse_credit_rating,
                       'status': self.parse_status,
                       'one': self.ones,
                       'actual_interest_rate': self.actual_interest_rate,
                       'expected_interest_rate': self.expected_interest_rate,
                       'fico_range': self.parse_fico_range,
                       'monthly_income': self.parse_monthly_income,
                       'home_ownership': self.parse_home_ownership,
                       'description_length': self.description_length}
        dtypes = []
        for c in self.columns: dtypes.append((c, '>f4'))
        self.raw_data = raw_data
        self.data = numpy.zeros((len(raw_data),), dtype=dtypes)
        for c in self.columns:
            f = lambda: self.identity(raw_data, c)
            if c in normalizers:
                f = lambda: normalizers[c](raw_data, c)
            self.data[c] = f()

    def description_length(self, d, col):
        return map(lambda x: len(x), d['loan_description'])

    def parse_home_ownership(self, d, col):
        return map(lambda x: 0 if x == 'RENT' else 1, d[col])

    def parse_monthly_income(self, d, col):
        return map(lambda x: min(x, 100000), d[col])

    def parse_fico_range(self, d, col):
        x = numpy.zeros(len(d[col]))
        for i, s in enumerate(d[col]):
            try:
                x[i] = int(s[0:3])
            except ValueError:
                x[i] = 660 # assume missing value / bad data is lowest possible credit score
        return x

    def actual_interest_rate(self, d, col):
        is_default = self.parse_status(d, 'status')
        return is_default * d['interest_rate']

    def expected_interest_rate(self, d, col):
        p_success = self.parse_status(d, 'status')
        return p_success * d['interest_rate']

    def ones(self, d, col):
        return map(lambda x: 1, d['interest_rate'])

    def parse_date(self, d, col):
        return map(lambda x: time.mktime(x.timetuple()), d[col])

    def parse_credit_rating(self, d, col):
        return map(lambda x: ((ord(x[0]) - ord('A')) * 5) + int(x[1]), d[col])

    def parse_status(self, d, col):
        def loan_status_collection_probability(status):
            # see https://www.lendingclub.com/info/statistics-performance.action for numbers
            if status == 'Fully Paid':
                return 1
            elif status == 'Charged Off':
                return 0
            elif status == 'In Grace Period':
                return 0.84
            elif status == 'Late (16-30 days)':
                return 0.77
            elif status == 'Late (31-120 days)':
                return 0.53
            elif status == 'Default':
                return 0.04
            elif status == 'Performing Payment Plan':
                return 0.5 # this status not listed, 50% is a guess
            raise Exception("Unknown status %s" % (status))

        p_return = []
        for i, status in enumerate(d[col]):
            p_r = None
            if status == 'Current':
                T = d['amount_funded_by_investors'][i]
                t = d['payments_to_date'][i]
                percent_remaining = 1 if T == 0 else (T-t)/T # TODO: how can T be zero?
                avg_default_rate = 0.07
                expected_default_rate = avg_default_rate * percent_remaining
                expected_default_rate = max(0, expected_default_rate)
                p_r = 1 - expected_default_rate
            else:
                p_r = loan_status_collection_probability(status)
            p_return.append(p_r)
        return p_return

    def identity(self, d, col):
        return d[col]


class LcPlotter:

    def __init__(self, raw_data, normalized_data, features, targets):
        self.features = features
        self.raw_data = raw_data.copy()
        self.normalized_data = normalized_data.copy()
        self.targets = targets
        subprocess.call(["mkdir", "plots"])

    def plot_correlations(self):
        smoothing_window = max(1, int(len(self.raw_data[self.targets[0]]) / 10))
        for c_f in self.features:            
            grouped_features = mlab.rec_groupby(self.normalized_data, [c_f], [(c_f, len, 'count')])
            is_discrete = len(grouped_features) < 100
            if c_f in self.raw_data:
                self.raw_data.sort(order=c_f)
            is_date = c_f.find('date') >= 0
            if is_date:
                self.raw_data.sort(order=c_f)
            else:
                self.normalized_data.sort(order=c_f)
            for c_t in self.targets:
                try:
                    f = plt.figure()
                    if is_discrete:
                        d = mlab.rec_groupby(self.normalized_data, [c_f], [(c_t, numpy.average, 'avg')])
                        plt.bar(d[c_f], d['avg'])
                    else:
                        y = None
                        if c_t in self.raw_data and self.raw_data[c_t].dtype == '>f4':
                            y = self.raw_data[c_t]
                        else:
                            y = self.normalized_data[c_t]        
                            convolved_y = numpy.convolve(numpy.ones(smoothing_window, 'd')/smoothing_window, y, mode='valid')
                            x = self.raw_data[c_f] if is_date else self.normalized_data[c_f]
                            plt.plot(x[0:convolved_y.shape[0]], convolved_y)
                            if is_date: f.autofmt_xdate()
                    plt.ylabel(c_t)
                    plt.xlabel(c_f)
                    plt.savefig("%s/%s_x_%s" % ('plots', c_f, c_t))
                except:
                    print "Error creating plot (%s, %s)" % (c_f, c_t)


class LcData:

    def __init__(self):
        self.csv_columns = ["loan_id","amount_requested","amount_funded_by_investors","interest_rate","loan_length","application_date","application_expiration_date","issued_date","credit_grade","loan_title","loan_purpose","loan_description","monthly_payment","status","total_amount_funded","debt_to_income_ratio","remaining_principal_funded_by_investors","payments_to_date_funded_by_investors_","remaining_principal_","payments_to_date","screen_name","city","state","home_ownership","monthly_income","fico_range","earliest_credit_line_date","open_credit_lines","total_credit_lines","revolving_credit_balance","revolving_line_utilization","inquiries_in_the_last_6_months","accounts_now_delinquent","delinquent_amount","delinquencies__last_2_yrs_","months_since_last_delinquency","public_records_on_file","months_since_last_record","education","employment_length","code"]


    def load_csv(self, fname):
        def clean_csv():
            print "Reading csv from file %s" % (fname)
            reader = csv.reader(open(fname, 'rb'))
            cleaned_fname = "/tmp/lc-%s.csv" % (random.random())
            print "Cleaning csv file using python csv library, writing new file to %s" % (cleaned_fname)
            writer = csv.writer(open(cleaned_fname, 'wb'))
            for i, row in enumerate(reader):
                # skip first 2 rows
                if i < 2: continue
                if len(self.csv_columns) == len(row):
                    writer.writerow(row)
                else:
                    print "\tError row %d, line contents:\"%s\"" % (i, ", ".join(row))
            return cleaned_fname

        cleaned_fname = clean_csv()
        converterd = {'interest_rate': fieldparsers.strip_non_numeric_and_parse,
                      'loan_length': fieldparsers.strip_non_numeric_and_parse,
                      'employment_length': fieldparsers.parse_employment_years,
                      'debt_to_income_ratio': fieldparsers.strip_non_numeric_and_parse,
                      'revolving_line_utilization': fieldparsers.strip_non_numeric_and_parse,
                      'status': fieldparsers.parse_status
                      }
        print "Loading csv via mlab"
        self.data = mlab.csv2rec(cleaned_fname, skiprows=2, converterd=converterd, names=self.csv_columns)
        subprocess.call(["rm", "-rf", cleaned_fname])
        print "Done."

    def exclude_values(self, col, values):
        indexes = numpy.where(numpy.all([self.data[col] != v for v in values], axis=0))
        return self.data[indexes]

def run(filename):
    lc_data = LcData()
    lc_data.load_csv(filename)
    status_types_to_exclude = ['Issued', 'In Review', 'Current']
    csv_data = lc_data.exclude_values('status', status_types_to_exclude)
    print "Removed status types [%s], num rows resulting: %d" % (", ".join(status_types_to_exclude), csv_data.shape[0])
    csv_data.sort(order='application_date')
    lc_data_features = LcDataExtractedFeatures()
    lc_data_features.create(csv_data)
    targets = ['status', 'expected_interest_rate', 'interest_rate']
    features = ['one', 'amount_requested', 'interest_rate', 'application_date', 'credit_grade', 'debt_to_income_ratio', 'monthly_income', 'fico_range', 'open_credit_lines', 'total_credit_lines', 'earliest_credit_line_date', 'home_ownership', 'description_length']
    plotter = LcPlotter(csv_data, lc_data_features.data, features, targets)
    plotter.plot_correlations()
    lc_learner = LcLearner(lc_data_features.data, csv_data)
    lc_learner.evaluate_all(features, 'status')

if __name__ == '__main__':
    run(sys.argv[1])



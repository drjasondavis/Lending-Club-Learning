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

class BinnedClassifier:
    
    def __init__(self, csv_col=None):
        self.csv_col = csv_col
    
    def get_bins(self, A_csv):
        return A_csv[self.csv_col] if self.csv_col != None else numpy.ones(len(b))

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
        print self.averages
        return self

    def predict(self, A, A_csv):
        bins = self.get_bins(A_csv)
        return map(lambda x: self.averages.setdefault(x, 0), A_csv[self.csv_col])
    

class ProbabilisticClassifier:

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        
    def get_classifier_probabilities(self, A):
        probabilities = self.base_classifier.predict_proba(A)
        return probabilities[:,0]

    def convert_to_prob(self, x):
        return self.prob_map[max(0, min(100, int(x * 100)))]

    def predict(self, A, A_csv):
        preds = self.get_classifier_probabilities(A)
        #return numpy.ones(len(preds))
        return map(self.convert_to_prob, preds.T)        

    def fit(self, A, A_csv, b):
        self.base_classifier.fit(A, b)
        preds = self.get_classifier_probabilities(A)
        counts = numpy.zeros((101))
        self.prob_map = numpy.zeros((101))
        prob_sums = numpy.zeros((101))
        for i, p in enumerate(preds):
            p = max(0, min(100, int(p * 100)))
            counts[p] += 1
            prob_sums[p] += b[i]
            
        min_support = min(1000, len(b))
        for i in range(101):
            support, prob_sum, j = 0, 0, 0
            while support < min_support:
                j += 1
                ind = i + (int(j/2) * (1 if j % 2 == 1 else -1))
                if ind > 100 or ind < 0: continue
                support += counts[ind]
                prob_sum += prob_sums[ind]
            self.prob_map[i] = prob_sum / support

        plt.figure()
        plt.plot(range(0,101,1), self.prob_map)
        plt.savefig('/tmp/plot.png')
        return self

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
        
        def annual_income(interest_rate_in_percent, amount, num_months):
            print interest_rate_in_percent[0:10]
            num_years = num_months / 12
            if numpy.min(num_months) < 12:
                raise Exception("Invalid data, num months: " + str(num_months))
            if numpy.average(interest_rate_in_percent) < 4:
                raise Exception("Invalid data, interest_rate_in_percent: " + str(interest_rate_in_percent))
            return (interest_rate_in_percent / 100) * (amount / num_years)
        
        loan_values = numpy.zeros((len(b_test), ), dtype=[('pred_irate', '>f4'), ('rand', '>f4'), ('actual_irate', '>f4'), ('irate', '>f4')])
        
        loan_values['pred_irate'] = -1 * A_test_csv['interest_rate'] * preds / 100
        loan_values['rand'] = numpy.random.rand(len(b_test))        
        loan_values['actual_irate'] = A_test_csv['interest_rate'] * b_test / 100        
        loan_values['irate'] = A_test_csv['interest_rate']
        loan_values.sort(order='pred_irate')
        loan_values['pred_irate'] *= -1

        amount_to_invest = 1000
        total_invested = 0
        investment_return = 0
        investment_per_loan = 25
        max_interest_rate = 25
        counts = []
        returns = numpy.zeros(len(loan_values['actual_irate']))
        for i, actual_return in enumerate(loan_values['actual_irate']):            
            if loan_values['irate'][i] > max_interest_rate: continue
            returns[i] = actual_return if i == 0 else actual_return + returns[i-1]
            if total_invested < amount_to_invest: 
                investment_return += investment_per_loan * loan_values['actual_irate'][i]
                total_invested += investment_per_loan
        
        returns = returns / (1 + numpy.array(range(len(returns))))
        plt.figure()
        #plt.plot(loan_values['pred_irate'], loan_values['actual_irate'])
        plt.plot(range(len(returns)), returns)
        #plt.hist(preds)
        plt.savefig('/tmp/plots/' + ".".join(cols) + '.png')

        avg_error = numpy.sum(abs(errors)) / errors.shape[0]
        return investment_return


    def evaluate_all(self, cols, target_col):
        print "Error binning by credit grade: %f\n" % (self.evaluate([], target_col, BinnedClassifier(csv_col='credit_grade')))
        def create_probabilistic_logistic_classifier():
            clf = sklearn.linear_model.LogisticRegression(C=10000, penalty='l1', scale_C=True)
            return ProbabilisticClassifier(clf)
        cv_plc = lambda x: self.evaluate(x, target_col, create_probabilistic_logistic_classifier())
        print "Error with all cols: %f\n" % (cv_plc(cols))
        for c in cols:            
            cols_copy = copy.copy(cols)
            cols_copy.remove(c)
            print "Column %s\n\twith only: %f\n\twithout: %f\n" % (c, cv_plc([c]), cv_plc(cols_copy))

        
    
class LcDataExtractedFeatures:
    
    def create(self, raw_data):
        self.columns = ['amount_requested', 'interest_rate', 'loan_length', 'application_date', 'credit_grade', 'status', 'one', 'actual_interest_rate', 'debt_to_income_ratio','monthly_income', 'fico_range', 'open_credit_lines', 'total_credit_lines', 'earliest_credit_line', 'home_ownership']
                        
        normalizers = {'application_date': self.parse_date,
                       'earliest_credit_line': self.parse_date,
                       'credit_grade': self.parse_credit_rating,
                       'status': self.parse_status,
                       'one': self.ones,
                       'actual_interest_rate': self.actual_interest_rate,
                       'fico_range': self.parse_fico_range,
                       'monthly_income': self.parse_monthly_income,
                       'home_ownership': self.parse_home_ownership}
        dtypes = []
        for c in self.columns: dtypes.append((c, '>f4'))
        self.raw_data = raw_data
        self.data = numpy.zeros((len(raw_data),), dtype=dtypes)
        for c in self.columns:
            f = lambda: self.identity(raw_data, c)
            if c in normalizers:
                f = lambda: normalizers[c](raw_data, c)
            self.data[c] = f()
        
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
                
        print mlab.rec_groupby(d, [col], [(col, len, 'count')])
        p_return = []
        for i, status in enumerate(d[col]):
            if status == 'Current':
                T = d['amount_funded_by_investors'][i]
                t = d['payments_to_date'][i]
                percent_remaining = (T-t)/T
                avg_default_rate = 0.07
                expected_default_rate = avg_default_rate * percent_remaining
                expected_default_rate = max(0, expected_default_rate)                
                p_return.append(1 - expected_default_rate)
            else:
                p_return.append(loan_status_collection_probability(status))
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
        smoothing_window = max(1, int(len(self.raw_data[self.targets[0]]) / 50))
        for c_f in self.features:
            if c_f in self.raw_data:
                self.raw_data.sort(order=c_f)
            self.normalized_data.sort(order=c_f)
            for c_t in self.targets:
                y = None
                if c_t in self.raw_data and self.raw_data[c_t].dtype == '>f4':
                    y = self.raw_data[c_t]
                else:
                    y = self.normalized_data[c_t]
                plt.figure()
                convolved_y = numpy.convolve(y, numpy.ones(smoothing_window, 'd')/smoothing_window, 'same')
                plt.plot(self.normalized_data[c_f], convolved_y)
                plt.savefig("%s/%s_x_%s" % ('plots', c_f, c_t))
                
            
class LcData:

    def __init__(self):
        self.csv_columns = ["loan_id","amount_requested","amount_funded_by_investors","interest_rate","loan_length","application_date","application_expiration_date","issued_date","credit_grade","loan_title","loan_purpose","loan_description","monthly_payment","status","total_amount_funded","debt_to_income_ratio","remaining_principal_funded_by_investors","payments_to_date_funded_by_investors_","remaining_principal_","payments_to_date","screen_name","city","state","home_ownership","monthly_income","fico_range","earliest_credit_line","open_credit_lines","total_credit_lines","revolving_credit_balance","revolving_line_utilization","inquiries_in_the_last_6_months","accounts_now_delinquent","delinquent_amount","delinquencies__last_2_yrs_","months_since_last_delinquency","public_records_on_file","months_since_last_record","education","employment_length","code"]


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
                    print "\tError row %d: %s " % (i, ", ".join(row))
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
    csv_data = lc_data.exclude_values('status', ['Issued', 'In Review'])
    csv_data.sort(order='application_date')
    lc_data_features = LcDataExtractedFeatures()
    lc_data_features.create(csv_data)
    targets = ['status']
    features = ['one', 'amount_requested', 'interest_rate', 'application_date', 'credit_grade', 'debt_to_income_ratio', 'monthly_income', 'fico_range', 'open_credit_lines', 'total_credit_lines', 'earliest_credit_line', 'home_ownership']
    plotter = LcPlotter(csv_data, lc_data_features.data, features, targets)
    plotter.plot_correlations()
    lc_learner = LcLearner(lc_data_features.data, csv_data)
    lc_learner.evaluate_all(features, 'status')

if __name__ == '__main__': 
    run(sys.argv[1])
        
    

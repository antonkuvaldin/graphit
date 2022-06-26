from . import Report
from collections import OrderedDict

import statsmodels
import pandas
import pickle


class PandasReport(Report):
    def parse_hierarchy_dict(self, dictionary: dict):
        new_dict = {(level1_key, level2_key, level3_key): [values]
                    for level1_key, level2_dict in dictionary.items()
                    for level2_key, level3_dict in level2_dict.items()
                    for level3_key, values in level3_dict.items()}
        new_dict = OrderedDict(new_dict)
        return new_dict

    def calc_conf_interval(self, distr, alpha=0.95):
        a, b = statsmodels.stats.api.DescrStatsW(distr).tconfint_mean(alpha= 1 -alpha)
        m = distr.mean()
        err = max( m -a, b- m)
        # norm_rv = stats.norm()
        # z_crit = norm_rv.ppf(1 - alpha / 2)
        # mu_hat = numpy.mean(distr)
        # var_hat = numpy.var(distr)
        # n = len(distr)
        # err = z_crit * numpy.sqrt(var_hat / n)
        # left = mu_hat - z_crit * numpy.sqrt(var_hat / n)
        # right = mu_hat + z_crit * numpy.sqrt(var_hat / n)
        return m, err

    @staticmethod
    def find_more_values(s, baseline):
        if s.name[1] != 'value':
            return ['' for _ in s]
        for k, v in s.to_dict().items():
            if k == baseline:
                baseline_val = v
                break
        if s.name[0] in ['MAE', 'MSE']:
            good = 'background-color: #ECA6A6'
            bad = 'background-color: #B4CFB0'
        else:
            good = 'background-color: #B4CFB0'
            bad = 'background-color: #ECA6A6'
        color = []
        for k, v in s.to_dict().items():
            if k == baseline:
                color.append('')
                continue
            if v > baseline_val:
                color.append(good)
            elif v < baseline_val:
                color.append(bad)
            elif v == baseline_val:
                color.append('background-color: #E2DEA9')
            else:
                color.append('')
        return color

    @staticmethod
    def find_intervals(s, metrics, val, in_color, out_color):
        if s.name[0] not in metrics or s.name[1] != 'confidence':
            return ['' for _ in s]
        color = []
        for k, v in s.to_dict().items():
            interval = v.split(' ')
            try:
                left, right = interval
                left = float(left)
                right = float(right)
            except Exception:
                color.append(out_color)
                continue
            if val > left and val < right:
                color.append(in_color)
            else:
                color.append(out_color)
        return color

    @staticmethod
    def find_interval_higher(s, metrics, val, less_color, high_color):
        if s.name[0] not in metrics or s.name[1] != 'confidence':
            return ['' for _ in s]
        color = []
        for k, v in s.to_dict().items():
            interval = v.split(' ')
            try:
                left, right = interval
                left = float(left)
                right = float(right)
            except Exception:
                color.append(less_color)
                continue
            if val > left and val > right:
                color.append(high_color)
            else:
                color.append(less_color)
        return color

    def show(self, style=True, baseline=None):
        results = pandas.DataFrame(self.parse_hierarchy_dict(self._results)).stack(level=[0])
        results = results.droplevel(0, axis=0)
        cols_order = []
        for metric in self._metrics:
            cols_order.append((metric.name, 'value'))
            cols_order.append((metric.name, 'confidence'))
        results = results[cols_order]
        if not style:
            return results
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        index_names = {
            'selector': '.index_name',
            'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
        }
        headers = {
            'selector': 'th:not(.index_name)',
            'props': 'background-color: #000066; color: white;'
        }
        results = results.style.set_table_styles([cell_hover, index_names, headers])

        results = results.set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
        ], overwrite=False)

        borders_style = dict()
        for indices in cols_order:
            borders_style[indices] = [{'selector': 'th', 'props': 'border-left: 1px solid white'},
                                      {'selector': 'td', 'props': 'border-left: 1px solid #000066'}]
        results.set_table_styles(borders_style, overwrite=False, axis=0)
        results.columns.names = ['Metric:', 'Result:']
        num_formats = dict()
        for col in results.data.columns:
            if col[1] != 'value':
                continue
            new_f = '{:.2f}'
            if results.data[col].abs().max() > 99:
                new_f = '{:.2e}'
            num_formats[col] = new_f
        results = results.format(num_formats)

        if baseline is not None:
            results = results.apply(lambda x: self.find_more_values(x, baseline=baseline), axis=0)
        results.apply(lambda x: self.find_intervals(x, ['Pearson', 'Kendall', 'Spearman'], 0, 'background-color: #B4CFB0', 'background-color: #ECA6A6'))
        results.apply(lambda x: self.find_interval_higher(x, ['R2'], 0, 'background-color: #ECA6A6', 'background-color: #B4CFB0'))

        # if baseline is not None:
        #
        #     results.style.apply(highlight_max, props='color:red;', axis=0, subset=slice_)
        return results

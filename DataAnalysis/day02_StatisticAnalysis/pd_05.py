# 使用Python进行假设检验
import pandas as pd
grades_df = pd.read_csv('..//data//grades.csv')
# print(grades_df.tail(3))
# print(grades_df.info())
early_submission = grades_df[grades_df['assignment1_submission'] <= '2015-12-31']
# print(type(early_submission))
# print(early_submission.head())
later_submisson = grades_df[grades_df['assignment1_submission'] > '2015-12-31']
# print('提前提交的均值：', early_submission.mean())
# print('延迟提交的的均值：', later_submisson.mean())

from scipy import stats
# 使用t检验来比较两个总体是否有显著差异，即：早提交作业的学生与晚提交作业的学生成绩是否有显著差异
# 零假设：早提交的学生总体和晚提交的学生总体没有显著差异
# 备择假设：两个总体有显著差异
# 构造一个与此相关的统计量，如果该统计量非常的大（即已经超过了一定的临界值），即p-value<alpha
# 则可以认为这种差异并不仅仅是由抽样误差带来的，因此我们可以拒绝原假设，认为两个总体有显著差异。

# pvalue=0.16，表示两个总体在assignment1上的成绩没有显著差异的概率是0.16>0.05，小概率事件没有发生，不能拒绝原假设
print(stats.ttest_ind(early_submission['assignment1_grade'], later_submisson['assignment1_grade']))
# pvalue=0.18，表示两个总体在assignment1上的成绩没有显著差异的概率是0.16>0.05，小概率事件没有发生，不能拒绝原假设
print(stats.ttest_ind(early_submission['assignment2_grade'], later_submisson['assignment2_grade']))
# pvalue=0.99，表示两个总体在assignment1上的成绩没有显著差异的概率是0.16>0.05，小概率事件没有发生，不能拒绝原假设
print(stats.ttest_ind(early_submission['assignment6_grade'], later_submisson['assignment6_grade']))


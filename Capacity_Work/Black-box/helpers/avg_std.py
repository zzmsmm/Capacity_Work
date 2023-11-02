import numpy as np

# situation_1
'''
result = "90.395%90.755%90.495%90.214%90.264%90.475%90.224%90.204%90.104%90.014%"
auc_list = result.split('%')[0:10]
auc_list_float = [float(x) for x in auc_list]

auc_mean = np.mean(auc_list_float)
auc_std = np.std(auc_list_float)

print(str(round(auc_mean, 3)) + '%')
print(str(round(auc_std, 3)) + '%')
'''

# situation_2
# '''
result = "99.609% (0.000) / 90.815%99.609% (0.000) / 90.525%100% (0.000) / 90.525%99.609% (0.000) / 90.184%100% (0.000) / 90.425%100% (0.000) / 90.755%100% (0.000) / 90.294%100% (0.063) / 90.935%100% (0.000) / 90.144%100% (0.000) / 90.274%"

wm_auc_list = []
auc_list = []
wm_loss_list = []

for i in range(0, 11):
    if i == 0:
        wm_auc_list.append(result.split('/ ')[i].split('%')[0])
        wm_loss_list.append(result.split('/ ')[i].split('%')[1].split('(')[1].split(')')[0])
    elif i < 10:
        wm_auc_list.append(result.split('/ ')[i].split('%')[1])
        wm_loss_list.append(result.split('/ ')[i].split('%')[2].split('(')[1].split(')')[0])
        auc_list.append(result.split('/ ')[i].split('%')[0])
    else:
        auc_list.append(result.split('/ ')[i].split('%')[0])

wm_auc_list_float = [float(x) for x in wm_auc_list]
wm_loss_list_float = [float(x) for x in wm_loss_list]
auc_list_float = [float(x) for x in auc_list]

wm_auc_mean = np.mean(wm_auc_list_float)
wm_loss_mean = np.mean(wm_loss_list_float)
auc_mean = np.mean(auc_list_float)

wm_auc_std = np.std(wm_auc_list_float)
wm_loss_std = np.std(wm_loss_list_float)
auc_std = np.std(auc_list_float)

print(str('%.1f' % wm_auc_mean) + '% (' + str('%.3f' % wm_loss_mean) + ') / ' + str('%.3f' % auc_mean) + '%')
print(str('%.2f' % wm_auc_std) + '% (' + str('%.3f' % wm_loss_std) + ') / ' + str('%.3f' % auc_std) + '%')
# '''

# situation_3
'''
result = "25% (2.689)19% (4.157)31% (3.134)17% (4.677)19% (3.571)18% (4.492)17% (3.922)18% (4.135)24% (3.401)18% (4.667)"

wm_auc_list = []
wm_loss_list = []

for i in range(0, 11):
    if i == 0:
        wm_auc_list.append(result.split('(')[i].split('%')[0])
    elif i < 10:
        wm_auc_list.append(result.split('(')[i].split(')')[1].split('%')[0])
        wm_loss_list.append(result.split('(')[i].split(')')[0])
    else:
        wm_loss_list.append(result.split('(')[i].split(')')[0])

wm_auc_list_float = [float(x) for x in wm_auc_list]
wm_loss_list_float = [float(x) for x in wm_loss_list]

wm_auc_mean = np.mean(wm_auc_list_float)
wm_loss_mean = np.mean(wm_loss_list_float)

wm_auc_std = np.std(wm_auc_list_float)
wm_loss_std = np.std(wm_loss_list_float)

print(str('%.1f' % wm_auc_mean) + '% (' + str('%.3f' % wm_loss_mean) + ')')
print(str('%.2f' % wm_auc_std) + '% (' + str('%.3f' % wm_loss_std) + ')')
'''
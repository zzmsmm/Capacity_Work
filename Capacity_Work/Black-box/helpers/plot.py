import matplotlib.pyplot as plt
plt.switch_backend('agg')

dataset = 'cifar10'
Watermark = ['content', 'noise', 'frontier-stitching', 'exponential-weighting']
Attack = ['AO', 'FT', 'NP']
L = ['256', '512', '1024', '2048', '4096', '8192']
N = ['1', '2', '3', '4',  '5']

for watermark in Watermark:
    for attack in Attack:
        for k in N:
            for i in L:
                if attack == 'AO':
                    file = f'../capacity_log/{dataset}_resnet50/{watermark}/{attack}/{k}/{dataset}_resnet50_{watermark}_{k}_L{i}_ao.txt'
                else:
                    file = f'../capacity_log/{dataset}_resnet50/{watermark}/{attack}/{k}/{dataset}_resnet50_{watermark}_{k}_L{i}.txt'
                x = []
                test_acc = []
                wm_acc = []
                with open(file, 'r') as f:
                    for line in f:
                        words = line.split()
                        if attack == 'NP':
                            if words[0] == '0.20':
                                x.append(0)
                                test_acc.append(float(words[1]))
                                wm_acc.append(float(words[2]))
                            x.append(float(words[0]))
                            test_acc.append(float(words[1]))
                            wm_acc.append(float(words[2]))
                        else:
                            x.append(int(words[0])+1)
                            test_acc.append(float(words[1]))
                            wm_acc.append(float(words[2]))
                
                plt.plot(x, test_acc,label=f'test_acc')
                plt.plot(x, wm_acc,label=f'wm_acc')

                if attack == 'NP':
                    plt.xlabel("Pruning Rate")
                else:
                    plt.xlabel("Epoch")
                
                plt.ylabel("ACC")
                plt.title(f"{watermark}_{attack}_{dataset}_resnet50_1_L{i}")
                plt.legend()

                if attack == 'AO':
                    save_path = f'../capacity_log/{dataset}_resnet50/{watermark}/{attack}/{k}/{dataset}_resnet50_{watermark}_{k}_L{i}_ao.png'
                else:
                    save_path = f'../capacity_log/{dataset}_resnet50/{watermark}/{attack}/{k}/{dataset}_resnet50_{watermark}_{k}_L{i}.png'
                
                plt.show()
                plt.savefig(save_path)
                plt.close()

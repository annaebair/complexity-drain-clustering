import csv
import pandas as pd
import matplotlib.pyplot as plt


files = ['10mil_3_11_results.csv',
         '10mil_3_18_results_2_agents.csv',
         '10mil_3_11_results_4_agents.csv',
         '10mil_3_18_results_4_agents.csv',
         '10mil_3_11_results_8_agents.csv',
         '10mil_3_18_results_8_agents.csv',
         '10mil_3_25_results.csv',
         '10mil_3_26_results.csv'
         ]


seed_list = []
num_agents_list = []
k_plus_list = []
k_minus_list = []
gamma_list = []
fit_list = []
size_list = []

for file in files:
    with open(file) as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row:
                seed, gridsize, num_agents, iterations, k_plus, k_minus, gamma, fit, size, objects_dict = row

                seed = int(seed)
                num_agents = int(num_agents)
                k_plus = float(k_plus)
                k_minus = float(k_minus)
                gamma = int(gamma)
                fit = float(fit)
                size = float(size)

                if k_plus < 0.2:
                    seed_list.append(seed)
                    num_agents_list.append(num_agents)
                    k_plus_list.append(k_plus)
                    k_minus_list.append(k_minus)
                    gamma_list.append(gamma)
                    fit_list.append(fit)
                    size_list.append(size)


d = {'seed': seed_list,
     'num_agents': num_agents_list,
     'k_plus': k_plus_list,
     'k_minus': k_minus_list,
     'gamma': gamma_list,
     'fit': fit_list,
     'size': size_list
     }

df = pd.DataFrame(d)
df = df.groupby(['num_agents', 'k_plus', 'k_minus', 'gamma']).mean()
df = df.drop(columns=['seed'])
AGENTS = 8
df_by_agents = df.query(f'num_agents == {AGENTS}')
flag4 = True
flag8 = True
for i in df_by_agents.index:
    agents, kplus, kminus, _ = i
    gammas = df_by_agents.loc[(agents, kplus, kminus)]
    fit4, size4 = gammas.loc[4]
    fit8, size8 = gammas.loc[8]

    if fit4 > fit8:
        if flag4 is True:
            plt.scatter(kplus, kminus, c='blue', s=fit4 * 15, alpha=0.5, label='$\Gamma$=4 larger')
            flag4 = False
        else:
            plt.scatter(kplus, kminus, c='blue', s=fit4 * 15, alpha=0.5)

    else:
        if flag8 is True:
            plt.scatter(kplus, kminus, c='red', s=fit8 * 15, alpha=0.5, label='$\Gamma$=8 larger')
            flag8 = False
        else:
            plt.scatter(kplus, kminus, c='red', s=fit8 * 15, alpha=0.5)

plt.xlabel('k+')
plt.ylabel('k-')
plt.title(f'Fitness for {AGENTS} agents')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def sample(reward):
    noise = np.random.gumbel(size=len(reward))
    sample = np.argmax(reward + noise)
    return sample

n_samples_d=30
n_samples_p=10

r = [0, -0.075,-1]  
reward = np.array(r)
sum_exp= np.exp(r[0])+np.exp(r[1])+np.exp(r[2])
p_0 = np.exp(r[0])/sum_exp
p_1 = np.exp(r[1])/sum_exp
p_2 = np.exp(r[2])/sum_exp

print("ground truth probability of action 1 =", p_0)
print("ground truth probability of action 2 =", p_1)
print("ground truth probability of action 3 =", p_2)


def data_generate(reward):
    demonstrations=pd.DataFrame({'action': [sample(reward) for _ in range(n_samples_d)]})
    p_MLE=demonstrations.value_counts(sort=False)/n_samples_d
    reward_12=np.array([r[1], r[2]])
    reward_01=np.array([r[0], r[1]])
    reward_02=np.array([r[0], r[2]])
    preferences_12 = pd.DataFrame({'preferred': [sample(reward_12) for _ in range(n_samples_p)]})
    preferences_01 = pd.DataFrame({'preferred': [sample(reward_01) for _ in range(n_samples_p)]})
    preferences_02 = pd.DataFrame({'preferred': [sample(reward_02) for _ in range(n_samples_p)]})
    p_12=preferences_12.value_counts(sort=False)/n_samples_p
    p_01=preferences_01.value_counts(sort=False)/n_samples_p
    p_02=preferences_02.value_counts(sort=False)/n_samples_p
    return p_12,p_01,p_02,p_MLE

def equations(vars,weight=1):
    x, y,z = vars
    x, y,z = vars
    eq1 = weight*p_MLE[0]+p_02[0] + p_01[0]-x*(weight+1/(x+y)+1/(x+z))
    eq2 = weight*p_MLE[1]+p_12[0] + p_01[1]-y*(weight+1/(x+y)+1/(z+y))
#     eq1 = p_02[0] + p_01[0]-x*(1/(x+y)+1/(x+z))
#     eq2 = p_12[0] + p_01[1]-y*(1/(x+y)+1/(z+y))
    eq3 = 1-x-y-z
    return [eq1, eq2, eq3]


def equations_preferences(vars):
    x, y,z = vars
    x, y,z = vars
    eq1 = p_02[0] + p_01[0]-x*(1/(x+y)+1/(x+z))
    eq2 = p_12[0] + p_01[1]-y*(1/(x+y)+1/(z+y))
    eq3 = 1-x-y-z
    return [eq1, eq2, eq3]

# p_PD = np.empty(3)

# p_PD =  fsolve(equations, (1/3, 1/3, 1/3))
save_P=[]
save_D=[]
save_DP=[]
for i in range(10):
    p_12,p_01,p_02,p_MLE = data_generate(reward)
    p_DP = fsolve(equations, (1/3, 1/3, 1/3))
    p_P = fsolve(equations_preferences, (1/3, 1/3, 1/3))
    save_P.append(p_P)
    save_D.append(list(np.array(p_MLE)))
    save_DP.append(p_DP)
    del p_P,p_DP,p_12,p_01,p_02,p_MLE
result_P=np.array(save_P)
result_D=np.array(save_D)
result_DP=np.array(save_DP)

result_P_mean,result_P_std=np.mean(result_P,0),np.std(result_P,0)
result_D_mean,result_D_std=np.mean(result_D,0),np.std(result_D,0)
result_DP_mean,result_DP_std=np.mean(result_DP,0),np.std(result_DP,0)
# plt.bar(categories, means, color='blue', alpha=0.7, label='Mean')
# plt.errorbar(categories, means, yerr=errors, fmt='o', color='r', capsize=5)

categories=[r'$\tau_1$',r'$\tau_2$',r'$\tau_3$']
groundtruth=[p_0,p_1,p_2]
# plt.title("Demonstrtaion vs Groundtruth")
x = np.arange(len(categories))
width = 0.2  

plt.bar(x - width*1.5, groundtruth, width, label='Groundtruth')
plt.bar(x - width/2, result_D_mean, width, label='SFT')
plt.errorbar(x - width/2, result_D_mean, yerr=result_D_std, fmt='o', color='r', capsize=5)
plt.bar(x + width/2, result_P_mean, width, label='DPO')
plt.errorbar(x + width/2, result_P_mean, yerr=result_P_std, fmt='o', color='r', capsize=5)
plt.bar(x + width*1.5, result_DP_mean, width, label='AIHF')
plt.errorbar(x + width*1.5, result_DP_mean, yerr=result_DP_std, fmt='o', color='r', capsize=5)
x = range(3)
plt.xticks(x, categories)
plt.ylabel("Probability",fontsize=15)
plt.legend()
plt.savefig("Demonstrtaion vs Groundtruth.pdf",dpi=400)
plt.savefig("Demonstrtaion vs Groundtruth.png",dpi=400)
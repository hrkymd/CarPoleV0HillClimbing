import gym
import numpy as np
import csv

# ログファイル作成用

methodName = "CarPoleHillClimbing"
resultName = "success"
# resultName = "failed"
thetaType = "1"
# thetaType = "10"

srblist = []
r_bestFile = open(methodName + resultName + thetaType + ".csv", 'w')
srbWriter = csv.writer(r_bestFile)
srbWriter.writerow(["generation", "r_best"])

env = gym.make('CartPole-v0')

n = 4 # パラメータ数
T = 200 # 時刻上限
M = 5 #エピソード回数
alpha = 0.1

# カウント用変数
count = 0
success = 0
failed = 0

# 100試行
while(count < 100):
    print("count = {0}".format(count))

    theta = 2 * np.random.uniform(low = 0.0, high = 1.0, size = n) - 1
    # theta = 2 * np.random.uniform(low=0.0, high=1.0, size=n) - 10
    #print(theta)
    r_best = 0.0
    g = 0

    while(g < 2000):

        # 新しい政策を作る
        theta_new = theta + (alpha * (2 * np.random.uniform(low=0.0, high=1.0, size=n) - 1))
        # theta_new = theta + (alpha * (2 * np.random.uniform(low=0.0, high=1.0, size=n) - 10))
        r_total = 0.0

        for i_episode in range(M):
            observation = env.reset()
            for t in range(T):
                # env.render()
                thetaSt = theta_new.dot(observation.transpose())
                if thetaSt < 0.0:
                    action = 0
                else:
                    action = 1
                observation, reward, done, info = env.step(action)
                # 報酬の合計を計算
                r_total += reward

                if done:
                    break

        # count == 0の時のみログをとる
        if count == 0:
            srblist.append([g, r_best])

        # 報酬の合計がそれまでより高い場合、r_bestとthetaを更新
        if r_total > r_best :
            r_best = r_total
            theta = theta_new
            # r_bestが終了条件を満たす場合(成功の場合)、ループを抜ける
            if r_best >= 200.0 * M:
                print("成功")
                success += 1
                # count == 0の時のみログをとる
                if count == 0:
                    srblist.append([g, r_best])
                break

        g = g + 1
        #世代数 == 2000となると失敗
        if g == 2000:
            print("失敗")
            failed += 1

    print("最終政策:{0}".format(theta))
    count = count + 1

print("成功数 = {0}".format(success))
print("失敗数 = {0}".format(failed))

# ログの書き込み
srbWriter.writerows(srblist)
r_bestFile.close()
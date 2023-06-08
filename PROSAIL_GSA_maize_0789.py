import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import prosail
import random
random.seed()
#import pyprosail
#iterat      = 2000
iterat      =50000
################### IMPORT MODES AND SPECTRAL BANDS SENTINEL ########################

df = pd.read_csv('D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/maize/WQG09_ref_maize0789.csv')
traitrange = pd.read_csv('D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/maize/Prosailranges_maize0789.csv')

#df = pd.read_csv('D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/WQG09_ref_maize09.csv')
#traitrange = pd.read_csv('D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/Prosailranges_maize09.csv')


spectralsensitivityfile = 'D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/P4M-SRF.csv'
s2sens = pd.read_csv(spectralsensitivityfile)
traitslist = ['N','Cab','Car','Cbp','Cw','Cm','Psoil','LAI','hot','tts','tto','psi','view_azimuth','Bsoil']
straitslist = ['N', 'LAI', 'Bsoil','Cab','Car','Cm','Cw','Cbp','Psoil','hot']       #生物物理变量名称
bandslist = ['B1','B2','B3','B4', 'B5']#波段名称
s2min = np.array( [ 434, 544, 634, 714, 814])   #对应波段波长最小值
s2max = np.array( [ 466, 576, 666, 746, 866])  #对应波段波长波段最大值

Nth               = traitrange[traitrange['Symbol'] == 'N']
LAI               = traitrange[traitrange['Symbol'] == 'LAI']
#ALA               = traitrange[traitrange['Symbol'] == 'ALA']
CAB               = traitrange[traitrange['Symbol'] == 'Cab']
CAR               = traitrange[traitrange['Symbol'] == 'Car']
CDM               = traitrange[traitrange['Symbol'] == 'Cm']     #cm Leaf dry matter content
CW                = traitrange[traitrange['Symbol'] == 'Cw']
CBP               = traitrange[traitrange['Symbol'] == 'Cbp']
Bsoil             = traitrange[traitrange['Symbol'] == 'Bsoil']
Psoil             = traitrange[traitrange['Symbol'] == 'Psoil']
HOT             = traitrange[traitrange['Symbol'] == 'hot']
TTs             = traitrange[traitrange['Symbol'] == 'tts']
TTo             = traitrange[traitrange['Symbol'] == 'tto']
Psi             = traitrange[traitrange['Symbol'] == 'psi']
view_azimuth    = 302     #观察者方位角 虽然写上但是没什么用  为替换其他参数做准备

#-----------------------------------------------------------------------------------------------------------------------
rel_measured_data = df.iloc[:, 1:6].values
reflectance = pd.DataFrame(columns=('B1','B2','B3', 'B4', 'B5'))
#reflectance = pd.DataFrame([rel_measured_data[0]], columns=('B1','B2','B3', 'B4', 'B5'))

Cab_measured_data = df.iloc[:, 6:7].values
LAI_measured_data = df.iloc[:, 7:8].values


for i in range(len(rel_measured_data)):
    refl = pd.DataFrame([rel_measured_data[i]], columns=('B1','B2','B3', 'B4', 'B5'))
    reflectance = reflectance.append(refl)



x = float(Nth['Fix']), float(CAB['Fix']), float(CAR['Fix']), float(CBP['Fix']), float(CW['Fix']), float(CDM['Fix']), float(Psoil['Fix']), float(LAI['Fix']), float(HOT['Fix']), float(TTs['Fix']), float(TTo['Fix']), float(Psi['Fix']), view_azimuth, float(Bsoil['Fix'])
x = list(x)
################### RUN MODAL CURVATURE ########################
################################################################

N, chloro, caroten, brown, EWT, LMA, psoil, lai, hot_spot, tts, tto, psi, view_azimuth, bsoil = x
#pro_out = pyprosail.run(N, chloro, caroten, brown, EWT, LMA, psoil, lai, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth, LIDF)
# 如果typelidf=2， lidfa代表平均叶倾角 chloro代表Cab  caroten代表Car  brown代表Cbrown  EWT 代表Cw  LMA代表Cm
#rho_canopy= prosail.run_prosail(N, chloro, caroten, brown, EWT, LMA, lai, LIDF, hot_spot,
                             #tts, tto, psi,typelidf=2,
                              #rsoil = 1., psoil=psoil, factor="SDR")
rho_canopy = prosail.run_prosail(N, chloro, caroten, brown,  EWT, LMA, lai, -0.35, hot_spot,
                                 tts, tto,  psi, typelidf=1, lidfb=-0.15,
                        rsoil = bsoil, psoil=psoil, factor="SDR")
# 原始的代码是由pyprosail写的，，需要进行修改换到prosail底层，因为pyprosail不在更新了，，但是这两个函数的返回值又有不同，需要进行修改！！！
#plt.plot(np.arange(400,2501),rho_canopy,'r-')
#plt.show()
LL=len(rho_canopy)  # 求出LL2101  因为是从
#pro_out=tuple(2101,2)
pro_out= np.zeros((2101,2),dtype=float)   # 初始化一个tuple 进行赋值 下面这3,4行代码解决两个prosail函数不一样的问题
for i in range(LL):
    pro_out[i][0] =int(i+400)
    pro_out[i][1]=rho_canopy[i]
#s2out    = pd.DataFrame(columns={'B2','B3','B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'})  #哨兵2波段名称
#s2outt   = np.array([[490,560,665,705,740,783,865,1610,2190],np.zeros(9)])
################### RUN MIN&MAX CURVES ########################
s2out    = pd.DataFrame(columns={'B1','B2','B3','B4', 'B5'})      #P4M 多光谱包含5个波段
s2outt   = np.array([[450,560,650,730,840],np.zeros(5)])             #P4M 多光谱包含5个波段，每个波段的中心中心波长
for ib, band in enumerate(bandslist):
    weigthedvalue = 0
    bandname = 'S2A_SR_AV_' + band
    sumvalue        = np.sum(s2sens[bandname])
    for ix in np.where(s2sens[bandname] != 0)[0]:
        #iwl = np.where((pro_out[:,0]*1000) == int(s2sens['SR_WL'].iloc[ix]))[0][0]
        iwl = np.where((pro_out[:,0]) == int(s2sens['SR_WL'].iloc[ix]))[0][0]     #去掉乘以的1000，因为这个方法已经转换
        aaa=float(s2sens[bandname].iloc[ix])
        bbb=pro_out[iwl,1]
        weigthedvalue   = weigthedvalue + (float(s2sens[bandname].iloc[ix]) * pro_out[iwl,1])
    s2out['band'] = (weigthedvalue/sumvalue)
    s2outt[1,ib] =  (weigthedvalue/sumvalue)
#plt.plot(s2outt[0], s2outt[1])         #用固定值，得出的反射率，与光谱响应函数的乘积的和/光谱响应函数的和 这步类似于积分计算
#print("avgline")
#plt.show()       # 用固定值得到的反射率 经过了尺度变换
avgline = s2outt
###################################################################

# 用每一个生化参数的最大值最小值，这时用哪个值固定其它值，得出反射率，与光谱响应函数的乘积的和/光谱响应函数的和。得到两个14*9
minlines = np.empty((len(traitslist), 5))
maxlines = np.empty((len(traitslist), 5))
for iv, variable in enumerate(traitslist):
    if variable in straitslist:
        x = float(Nth['Fix']), float(CAB['Fix']), float(CAR['Fix']), float(CBP['Fix']), float(CW['Fix']), float(
            CDM['Fix']), float(Psoil['Fix']), float(LAI['Fix']), float(HOT['Fix']), float(TTs['Fix']), float(
            TTo['Fix']), float(Psi['Fix']), view_azimuth, float(Bsoil['Fix'])
        x = list(x)
        minvar = float(traitrange[traitrange['Symbol'] == variable]['Minimum'])
        maxvar = float(traitrange[traitrange['Symbol'] == variable]['Maximum'])
        x[iv] = minvar

        N, chloro, caroten, brown, EWT, LMA, psoil, lai, hot_spot, tts, tto, psi, view_azimuth, bsoil = x
        rho_canopy = prosail.run_prosail(N, chloro, caroten, brown, EWT, LMA, lai, -0.35, hot_spot,
                                         tts, tto, psi, typelidf=1, lidfb=-0.15,
                                         rsoil=bsoil, psoil=psoil, factor="SDR")
        # ---------------------------------------------解决两个函数不匹配的问题----------------------------------------------------------------------------------------------
        pro_out = np.zeros((2101, 2), dtype=float)  # 初始化一个tuple 进行赋值 下面这3,4行代码解决两个prosail函数不一样的问题
        for i in range(LL):
            pro_out[i][0] = int(i + 400)
            pro_out[i][1] = rho_canopy[i]
        s2out = pd.DataFrame(columns={'B1', 'B2', 'B3', 'B4', 'B5'})
        s2outt = np.array([[450, 560, 650, 730, 840], np.zeros(5)])  # P4M 多光谱包含5个波段，每个波段的中心中心波长

        for ib, band in enumerate(bandslist):
            weigthedvalue = 0
            bandname = 'S2A_SR_AV_' + band
            sumvalue = np.sum(s2sens[bandname])
            for ix in np.where(s2sens[bandname] != 0)[0]:
                iwl = np.where((pro_out[:, 0]) == int(s2sens['SR_WL'].iloc[ix]))[0][0]  # 去掉乘以的1000，因为这个方法已经转换
                weigthedvalue = weigthedvalue + (float(s2sens[bandname].iloc[ix]) * pro_out[iwl, 1])
            s2out['band'] = (weigthedvalue / sumvalue)
            s2outt[1, ib] = (weigthedvalue / sumvalue)
        plt.plot(s2outt[0], s2outt[1])
        #plt.show()
        minlines[iv] = s2outt[1]

        x[iv] = maxvar
        N, chloro, caroten, brown, EWT, LMA, psoil, lai, hot_spot, tts, tto, psi, view_azimuth, bsoil = x
        rho_canopy = prosail.run_prosail(N, chloro, caroten, brown, EWT, LMA, lai, -0.35, hot_spot,
                                         tts, tto, psi, typelidf=1, lidfb=-0.15,
                                         rsoil=bsoil, psoil=psoil, factor="SDR")
        # ---------------------------------------------解决两个函数不匹配的问题----------------------------------------------------------------------------------------------
        pro_out = np.zeros((2101, 2), dtype=float)  # 初始化一个tuple 进行赋值 下面这3,4行代码解决两个prosail函数不一样的问题
        for i in range(LL):
            pro_out[i][0] = int(i + 400)
            pro_out[i][1] = rho_canopy[i]
        s2out = pd.DataFrame(columns={'B1', 'B2', 'B3', 'B4', 'B5'})
        s2outt = np.array([[450, 560, 650, 730, 840], np.zeros(5)])  # P4M 多光谱包含5个波段，每个波段的中心中心波长

        for ib, band in enumerate(bandslist):
            weigthedvalue = 0
            bandname = 'S2A_SR_AV_' + band
            sumvalue = np.sum(s2sens[bandname])
            for ix in np.where(s2sens[bandname] != 0)[0]:
                iwl = np.where((pro_out[:, 0]) == int(s2sens['SR_WL'].iloc[ix]))[0][0]  # 去掉乘以的1000，因为这个方法已经转换
                weigthedvalue = weigthedvalue + (float(s2sens[bandname].iloc[ix]) * pro_out[iwl, 1])
            s2out['band'] = (weigthedvalue / sumvalue)
            s2outt[1, ib] = (weigthedvalue / sumvalue)
        plt.plot(s2outt[0], s2outt[1])
        plt.show()
        # print str(variable)
        print("variable：", variable)
        maxlines[iv] = s2outt[1]
############################################################################################################################
################## SET & RUN ITERATIONS ####################################################################################


total = pd.DataFrame(columns=('N', 'LAI', 'Bsoil','Cab','Car','Cm','Cw','Cbp','Psoil','hot', 'B1','B2','B3', 'B4', 'B5'))
nn=len(total)   #这里面nn等于0
row = 0
while len(total) < iterat:
        N           = np.random.uniform(Nth['Minimum'],Nth['Maximum'])
        chloro      = np.random.uniform(CAB['Minimum'],CAB['Maximum'])
        # 用高斯分布
        #chloro     = np.random.normal(21, 10)
        #if chloro<float(CAB['Minimum']):
        #    chloro=float(CAB['Minimum'])
        #elif chloro>float(CAB['Maximum']):
        #    chloro = float(CAB['Maximum'])
        #else:
        #  chloro = chloro
        EWT         = np.random.uniform(CW['Minimum'],CW['Maximum'])
        LMA         = np.random.uniform(CDM['Minimum'],CDM['Maximum'])
        brown       = np.random.uniform(CBP['Minimum'],CBP['Maximum'])
        caroten     = np.random.uniform(CAR['Minimum'],CAR['Maximum'])
        psoil       = np.random.uniform(Psoil['Minimum'],Psoil['Maximum'])
        lai         = np.random.uniform(LAI['Minimum'],LAI['Maximum'])
        #lai     = np.random.normal(3,1)
        #if lai<float(LAI['Minimum']):
        #    lai=float(LAI['Minimum'])
        #elif lai>float(LAI['Maximum']):
        #    lai = float(LAI['Maximum'])
        #else:
        #    lai = lai
        #用高斯分布
        hot_spot    = np.random.uniform(HOT['Minimum'],HOT['Maximum'])
        bsoil        = np.random.uniform(Bsoil['Minimum'],Bsoil['Maximum'])
        tts         = np.random.uniform(TTs['Minimum'], TTs['Maximum'])
        tto         = np.random.uniform(TTo['Minimum'], TTo['Maximum'])
        psi         = np.random.uniform(Psi['Minimum'], Psi['Maximum'])

        N=N[0]
        chloro=chloro[0]
        EWT=EWT[0]
        LMA=LMA[0]
        brown=brown[0]
        caroten=caroten[0]
        psoil=psoil[0]
        lai=lai[0]
        hot_spot=hot_spot[0]
        bsoil=bsoil[0]
        tts = tts[0]
        tto = tto[0]
        psi = psi[0]
        rho_canopy = prosail.run_prosail(N, chloro, caroten, brown, EWT, LMA, lai, -0.35, hot_spot,
                                         tts, tto, psi, typelidf=1, lidfb=-0.15,
                                         rsoil=bsoil, psoil=psoil, factor="SDR")

        # ---------------------------------------------解决两个函数不匹配的问题----------------------------------------------------------------------------------------------
        pro_out = np.zeros((2101, 2), dtype=float)  # 初始化一个tuple 进行赋值 下面这3,4行代码解决两个prosail函数不一样的问题
        for i in range(LL):
            pro_out[i][0] = int(i + 400)
            pro_out[i][1] = rho_canopy[i]

        s2out = pd.DataFrame(columns={'B1', 'B2', 'B3', 'B4', 'B5'})
        s2outt = np.array([[450, 560, 650, 730, 840], np.zeros(5)])  # P4M 多光谱包含5个波段，每个波段的中心中心波长

        for ib, band in enumerate(bandslist):
            weigthedvalue = 0
            bandname = 'S2A_SR_AV_' + band
            sumvalue        = np.sum(s2sens[bandname])
            for ix in np.where(s2sens[bandname] != 0)[0]:
                iwl = np.where((pro_out[:,0]) == int(s2sens['SR_WL'].iloc[ix]))[0][0]    #去掉乘以的1000，因为这个方法已经转换
                weigthedvalue   = weigthedvalue + (float(s2sens[bandname].iloc[ix]) * pro_out[iwl,1])
            s2out['band'] = (weigthedvalue/sumvalue)
            #rel_gauss=weigthedvalue / sumvalue
            #rel_gauss+=random.gauss(0, 0.04)    #加上高斯噪声4%
            #if rel_gauss<0:
            #   rel_gauss = weigthedvalue / sumvalue
            #else:
            #    rel_gauss=rel_gauss
            #s2outt[1,ib] = rel_gauss
            s2outt[1,ib] =  (weigthedvalue/sumvalue)
        #plt.plot(s2outt[0], s2outt[1])
        spectra = pd.DataFrame([s2outt[1]], columns=('B1','B2','B3', 'B4', 'B5'))
        traits  = pd.DataFrame([np.array([N, chloro, caroten, brown, EWT, LMA, psoil, lai, hot_spot, bsoil])], columns=('N', 'Cab','Car','Cbp','Cw','Cm','Psoil','LAI', 'hot', 'Bsoil'))
        total = total.append(pd.concat((traits,spectra),axis=1), ignore_index=True)
        row = row +1
        print("运行到第几行:",row)

###########------------基于L2范数损失函数，也被称为最小平方误差（LSE）--------------------------------------------------
#基于LSE代价函数进行真实反射率和模拟反射率判断
nCount=10 #取出前10%的最优值
Cab_pre_ave=np.zeros((len(rel_measured_data),1))
LAI_pre_ave=np.zeros((len(rel_measured_data),1))
for kk in range(len(rel_measured_data)):    #根据要输入的反射率的个数
    LSE_gather= np.zeros((iterat,2))
    for ii in range(iterat):
        simu_b1=total['B1'].iloc[ii]
        measured_b1=reflectance['B1'].iloc[kk]
        simu_b2=total['B2'].iloc[ii]
        measured_b2=reflectance['B2'].iloc[kk]
        simu_b3=total['B3'].iloc[ii]
        measured_b3=reflectance['B3'].iloc[kk]
        simu_b4=total['B4'].iloc[ii]
        measured_b4=reflectance['B4'].iloc[kk]
        simu_b5=total['B5'].iloc[ii]
        measured_b5=reflectance['B5'].iloc[kk]
        LSE=np.square(simu_b1-measured_b1)+np.square(simu_b2-measured_b2)+np.square(simu_b3-measured_b3)+np.square(simu_b4-measured_b4)+np.square(simu_b5-measured_b5)
        #LSE = (1/measured_b1)*np.square(simu_b1 - measured_b1) + (1/measured_b2)*np.square(simu_b2 - measured_b2) + (1/measured_b3)*np.square(
         #   simu_b3 - measured_b3) + (1/measured_b4)*np.square(simu_b4 - measured_b4) + (1/measured_b5)*np.square(simu_b5 - measured_b5)     #反演LAI加入权重
        LSE_gather[ii][0] = int(ii)
        LSE_gather[ii][1] = float(LSE)
    #代价函数结束
    #基于代价函数结果的逆序排序
    LSE_gather_XP=LSE_gather[np.lexsort(LSE_gather.T)]
    pred_index = np.zeros(nCount, dtype=int)  #index
    Cab_prediction=np.zeros(nCount,dtype=float)   #叶绿素
    LAI_prediction=np.zeros(nCount,dtype=float)   #LAI

    for jj in range(nCount):
        _index=int(LSE_gather_XP[jj][0])        #强制转换为int类型
        Cab_pre = total['Cab'].iloc[_index]    #求出最优的相对应的Cab等生理参量值
        LAI_pre = total['LAI'].iloc[_index]    # 求出最优的相对应的Cab等生理参量值
        pred_index[jj]=_index                   #求出指示的位置
        Cab_prediction[jj]=Cab_pre              #求出相对应值的叶绿素
        LAI_prediction[jj] = LAI_pre            # 求出相对应值的LAI

    Cab_pre_ave[kk][0] = np.average(Cab_prediction)
    LAI_pre_ave[kk][0] = np.average(LAI_prediction)
    print("第", kk, "个反射率的循环")
    #print("第", kk, "最优前10的CAB预测值的index：", pred_index)
    #print("第",kk,"最优前10的CAB预测值：",Cab_prediction)
    #print("第",kk,"最优前10的CAB预测值平均值：",np.average(Cab_prediction))
    #print("第",kk,"最优前10的LAI预测值：",LAI_prediction)
    #print("第",kk,"最优前10的LAI预测值平均值：",np.average(LAI_prediction))
    #print("第", kk, "----------------------------------------------结束")


#-----------Cab反演-----------------------------------------------------------------------------------------------------
test_y_cab20=Cab_measured_data[0:20]
test_y_cab40=Cab_measured_data[20:40]
test_y_cab60=Cab_measured_data[40:60]
test_predict_cab20=Cab_pre_ave[0:20]
test_predict_cab40=Cab_pre_ave[20:40]
test_predict_cab60=Cab_pre_ave[40:60]


test_y_cab=Cab_measured_data     #实测的cab
print("实测的Cab：", test_y_cab)
test_predict_cab=Cab_pre_ave     #模拟的cab
print("模拟的Cab：", test_predict_cab)
Y_cab = [20,80]
# 横轴数据。
X_cab = [20,80]

l=plt.scatter(test_y_cab20, test_predict_cab20,20,c='tomato',marker='o')          #西红柿红
s=plt.scatter(test_y_cab40, test_predict_cab40,20,c='limegreen',marker='o')      #浅绿
t=plt.scatter(test_y_cab60,test_predict_cab60,20,c='black',marker='o')            #黑色
plt.legend((l, s, t),('sep9_cab', 'Aug8_cab', 'jul7_cab',),scatterpoints=1,loc='upper left',ncol=3,fontsize=12)

#plt.plot(test_y_cab, test_predict_cab, '-')
plt.plot(X_cab,Y_cab,c='k',ls='-')
plt.xlabel(r"Validation Cab $[ug/cm^{2}]$")
plt.ylabel(r"Retrieved Can $[ug/cm^{2}]$")
plt.show()

test_y_ave_cab = np.average(test_y_cab)                              # 求测试数据集的平均值
test_predict_ave_cab = np.average(test_predict_cab)           #求预测数据集的平均值
aa1 = np.array(test_y_cab) - test_y_ave_cab                          #真实值-真实值平均值
aa2 = np.array(test_predict_cab) - test_predict_ave_cab       #预测值-预测值平均值
aa3 = test_predict_cab - test_y_cab[:len(test_predict_cab)]              # 真实值-预测值
ac = np.abs(test_predict_cab - test_y_cab[:len(test_predict_cab)])       # 预测值与真值的差的绝对值

bb1 = np.sum(aa1 * aa2)
cc1 = np.sqrt(np.sum(np.square(np.array(test_y_cab) - test_y_ave_cab)))
cc2 = np.sqrt(np.sum(np.square(np.array(test_predict_cab) - test_predict_ave_cab)))
r2 = np.square(bb1 / (cc1 * cc2))
print("Cab相关系数r2：", r2)

rr1=np.sum(np.square(aa3))
rr2 = np.sum(np.square(np.array(aa1)))
R2_Cab=1-rr1/rr2
print("Cab决定系数R2：", R2_Cab)

# RMSE 均方根误差
mse = (1/len(test_predict_cab))*np.sum(np.square(aa3))
rmse=np.sqrt(mse)
print("Cab均方根误差Rmse：", rmse)
# RPD 相对分析误差
RPD = np.sqrt(np.sum(np.square(aa1)) / np.sum(np.square(aa3)))
print("Cab相对分析误差RPD：", RPD)  # 4444444444444444444444444444444
# 平均绝对误差
MAE = (1/len(test_predict_cab))*np.sum(ac)
print("Cab平均绝对误差MAE：", MAE)  # 555555555555555555555555555555

#LAI反演#################--------------------------------------------------------------------------------------------

test_y_LAI20=LAI_measured_data[0:20]
test_y_LAI40=LAI_measured_data[20:40]
test_y_LAI60=LAI_measured_data[40:60]
test_predict_LAI20=LAI_pre_ave[0:20]
test_predict_LAI40=LAI_pre_ave[20:40]
test_predict_LAI60=LAI_pre_ave[40:60]


test_y_LAI=LAI_measured_data     #实测的LAI
print("实测的LAI：", test_y_LAI)
test_predict_LAI=LAI_pre_ave     #模拟的LAI
print("模拟的LAI：", test_predict_LAI)
Y_LAI = [1,7]
# 横轴数据。
X_LAI = [1,7]

l=plt.scatter(test_y_LAI20, test_predict_LAI20,20,c='tomato',marker='o')          #西红柿红
s=plt.scatter(test_y_LAI40, test_predict_LAI40,20,c='limegreen',marker='o')      #浅绿
t=plt.scatter(test_y_LAI60,test_predict_LAI60,20,c='black',marker='o')            #黑色
plt.legend((l, s, t),('sep9_LAI', 'Aug8_LAI', 'jul7_LAI',),scatterpoints=1,loc='upper left',ncol=3,fontsize=12)
plt.plot(X_LAI,Y_LAI,c='k',ls='-')

plt.xlabel(r"Validation LAI $[m^{2}m^{-2}]$")
plt.ylabel(r"Retrieved LAI $[m^{2}m^{-2}]$")
plt.show()

test_y_ave_LAI = np.average(test_y_LAI)                              # 求测试数据集的平均值
test_predict_ave_LAI = np.average(test_predict_LAI)           #求预测数据集的平均值
aa1 = np.array(test_y_LAI) - test_y_ave_LAI                          #真实值-真实值平均值
aa2 = np.array(test_predict_LAI) - test_predict_ave_LAI       #预测值-预测值平均值
aa3 = test_predict_LAI - test_y_LAI[:len(test_predict_LAI)]              # 真实值-预测值
ac = np.abs(test_predict_LAI - test_y_LAI[:len(test_predict_LAI)])       # 预测值与真值的差的绝对值

bb1 = np.sum(aa1 * aa2)
cc1 = np.sqrt(np.sum(np.square(np.array(test_y_LAI) - test_y_ave_LAI)))
cc2 = np.sqrt(np.sum(np.square(np.array(test_predict_LAI) - test_predict_ave_LAI)))
r2 = np.square(bb1 / (cc1 * cc2))
print("LAI相关系数r2：", r2)

rr1=np.sum(np.square(aa3))
rr2 = np.sum(np.square(np.array(aa1)))
R2_LAI=1-rr1/rr2
print("LAI决定系数R2：", R2_LAI)

# RMSE 均方根误差
mse = (1/len(test_predict_LAI))*np.sum(np.square(aa3))
rmse=np.sqrt(mse)
print("LAI均方根误差Rmse：", rmse)

# RPD 相对分析误差
RPD = np.sqrt(np.sum(np.square(aa1)) / np.sum(np.square(aa3)))
print("LAI相对分析误差RPD：", RPD)  # 4444444444444444444444444444444
# 平均绝对误差
MAE = (1/len(test_predict_LAI))*np.sum(ac)
print("LAI平均绝对误差MAE：", MAE)  # 555555555555555555555555555555

# ------------------------将得到反射率写入CSV---------------------------------------------------------------------------
# 只有Cab和LAI 同时满足
if R2_Cab >0.3 and R2_LAI > 0.3:
    csvfile = open(
        "D:/pycharmProject/prosail/SensitivityAnalysisPROSAIL-master/input/maize/WQG09_LAILCC_maize_0789.csv", 'a',newline='')
    # 2. 基于文件对象构建 csv写入对象
    writer = csv.writer(csvfile)
    content = ['Cab_measure', 'Cab_predict', 'LAI_measure', 'LAI_predict']
    # 3. 写入csv文件内容
    writer.writerow(content)

    for i in range(len(test_predict_cab)):
        Row_write_rel = []
        Row_write_rel.append(test_y_cab[i][0])
        Row_write_rel.append(test_predict_cab[i][0])
        Row_write_rel.append(test_y_LAI[i][0])
        Row_write_rel.append(test_predict_LAI[i][0])
        writer.writerow(Row_write_rel)
    # 4. 关闭文件
    csvfile.close()
    print("!!!文件写入完成")



###---------------------------------------------------------------------------------------------------------------------
pvalue = 0.01  #Pvalue值
r=0.55    #person相关系数r
xiv = 0
fig, ax = plt.subplots(5, 2, figsize=(20,10), sharey='col', sharex='all')
fig2, ax2 = plt.subplots(5, 2, figsize=(20,10), sharey='col', sharex='all')

for iv, variable in enumerate(traitslist):
        response = np.empty((1, 2*len(bandslist)))
        if variable in straitslist:
            if xiv > 4:
                riv = 1
            else:
                riv = 0
            if xiv > 4:
                civ = (xiv-6)
            else:
                civ = xiv
            xiv = xiv + 1
            bothlines = np.vstack((minlines[iv], maxlines[iv]))
            #bothlines = np.sort(bothlines, axis=0)
            for ib, band in enumerate(bandslist):
                a1=total[variable]
                a2=total[band]
                a3=scipy.stats.pearsonr(total[variable], total[band])
                a4=scipy.stats.pearsonr(total[variable], total[band])[0]
                a5=((scipy.stats.pearsonr(total[variable], total[band])[0]**2)**0.5)
                b1=scipy.stats.pearsonr(total[variable], total[band])[1]

                response[0,ib], response[0,ib+len(bandslist)] = ((scipy.stats.pearsonr(total[variable], total[band])[0]**2)**0.5), scipy.stats.pearsonr(total[variable], total[band])[1]  # **2代表求该值的平方，**0.5代表开平方
            ax[civ,riv].plot(s2outt[0], response[0,0:5], marker='D', markersize=5, linestyle='--')
            ax[civ,riv].title.set_text(str(variable))
            ax[civ,riv].set_ylim(0,1)
            ax[civ,riv].set_ylabel('Pearson`s R' )

            for ib, band in enumerate(bandslist):
                if response[0,(5+ib)] < pvalue and response[0,ib] >= r :
                    ax[civ,riv].axvspan(s2min[ib], s2max[ib], color='k',alpha=0.15)
            #        ax[0,1].fill_betweenx(1, s2min[ib], s2max[ib])

            ax2[civ,riv].fill_between(s2outt[0], bothlines[0], bothlines[1], alpha=0.2, interpolate=True)
            ax2[civ,riv].plot(s2outt[0], avgline[1], marker='D', markersize=5, linestyle='-')
            ax2[civ,riv].title.set_text(str(variable))
            ax2[civ,riv].set_ylim(0,0.4)
            ax2[civ,riv].set_ylabel('Reflectance')

plt.show()


print("运行结束")


import os
import numpy as np
from scipy.linalg import hadamard
import nibabel as nib
import scipy.io as sio
from datetime import datetime
from IPython.display import clear_output
import plotly.graph_objects as go
import matplotlib.pyplot as pp
import pandas as pd
from scipy.optimize import curve_fit

def set_protocol(numdelay,minPLD,totalLD,delaylin):
    '''
    This function allows you to set up GEeASL protocol specifications. 

    Inputs:
        - numdelay: numbers of delay/bolus blocks in Hadamard encoding, usually 3 or 7 for eASL
        - minPLD: the minimum PLD (s) after the final labelling bolus, specified as CV4, [0.7,4.0]
        - totalLD: the total labelling duration (s) of all labelling bolus, specified as CV5, [0.0,4.0]
        - delaylin: delay linearity factor, 0 for exponential T1-adj, 1 for linear, specified as CV7, [0.0,1.0]

    Outputs:
        protocol: a dictionary containing protocol specifications
    '''
    protocol = {}

    # protocol['name']: Protocol name
    protocol['name'] = f'GEeASL{numdelay:d}-CV4-{minPLD:.1f}-CV5-{totalLD:.1f}-CV7-{delaylin:.1f}'
    
    # protocol['had']: Hadamard encoding matrix
    protocol['had'] = hadamard(numdelay+1)[:,1:]
    nHadAcq = protocol['had'].shape[0]
    nHadBlock = protocol['had'].shape[1]
    
    # protocol['minPLD']: Mininum PLD (s)
    protocol['minPLD'] = minPLD
    
    # protocol['LDs']: Segmented LD series (s), calculated according to eASL manual
    T1a = 1.65
    LD_lin = np.repeat(totalLD/nHadBlock,nHadBlock)
    S_target = (1-np.exp(-totalLD/T1a))*np.exp(-minPLD/T1a)/nHadBlock
    PLD_exp = np.zeros(nHadBlock)
    LD_exp = np.zeros(nHadBlock)
    PLD_exp[0] = minPLD
    LD_exp[0] = -T1a*np.log(1-S_target*np.exp(PLD_exp[0]/T1a))
    for i in np.arange(1,nHadBlock):
        PLD_exp[i] = PLD_exp[i-1]+LD_exp[i-1]
        LD_exp[i] = -T1a*np.log(1-S_target*np.exp(PLD_exp[i]/T1a))
    LD_real = np.round((delaylin*LD_lin+(1-delaylin)*LD_exp),3)
    protocol['LDs'] = np.flip(LD_real,0)
    
    # protocol['TIs']: TIs required by oxford_asl 
    TI = np.zeros(nHadBlock)
    TI[0] = minPLD+LD_real[0]
    for i in np.arange(1,nHadBlock):
        TI[i] = TI[i-1]+LD_real[i]
    protocol['TIs'] = np.flip(TI,0)
    
    # protocol['PLDs']: PLDs required by basil
    protocol['PLDs'] = protocol['TIs'] - protocol['LDs']
    protocol['readout'] = 0.8
    protocol['scantime_1rpt'] = (totalLD+minPLD+protocol['readout'])*nHadAcq # time duration of 1 repeat
    
    # protocol['repeats']: Number of averages (NEX)
    protocol['scantimelim'] = 600
    protocol['repeats'] = 1
    # protocol['repeats'] = int((protocol['scantimelim']-np.mod(protocol['scantimelim'],protocol['scantime_1rpt']))/protocol['scantime_1rpt'])
    
    # protocol['scantime']: Total eASL scan duration (s)
    protocol['scantime'] = protocol['repeats']*protocol['scantime_1rpt']
    
    return protocol

def set_params(mode='rand'):
    '''
    This function allows you to set up other parameters in a cASL model.

    Outputs:
        params: a dictionary containing parameter specifications
    '''
    params = {}
    params['M_0a'] = 1
    params['lambda'] = 0.9
    params['T1_a'] = 1.65
    params['T1_t'] = 1.30
    params['labeleff'] = 0.85
    params['t'] = np.arange(0,10,0.001)
    params['batprior'] = 1.5
    params['batsdprior'] = 1.0

    params['SNR'] = 10
    params['dim'] = [100,100,1]
    # nSamp is used as the sampling number in CBF & ATT spaces in visualising protocol properties. 
    # The properties are calculated based on averages across nSamp samplings. 
    params['nSamp'] = 20
    params['fsltool'] = 'oxford_asl'

    params['nDelays'] = 7
    params['minPLDs'] = np.append(np.arange(0.1,1.5,0.1),np.arange(1.6,4.2,0.2))
    params['totalLDs'] = np.append(np.arange(0.2,2.2,0.2),np.arange(2.1,4.1,0.1))
    params['delaylins'] = np.arange(0.0,1.0+0.2,0.2)
    # params['minPLDs'] = [0.7,0.8]
    # params['totalLDs'] = [4.0]
    # params['delaylins'] = [0.0]
    params['scantimelim'] = 600
    params['blocktime'] = 40
    params['blocktaperL'] = 10
    params['blocktaperR'] = 10

    if mode == 'comb':
        params['mode'] = 'comb'
        params['cbfRange'] = np.arange(40,90,10)
        params['attRange'] = np.arange(0.5,3.0,0.5)
        params['abvRange'] = np.arange(0.0,2.5,0.5)
        params['atta_start'] = 0.0
        params['atta_step'] = 0.5
    if mode == 'rand':
        params['mode'] = 'rand'
        params['cbfRange'] = np.array([40,80])
        params['attRange'] = np.array([0.5,2.5])
        params['attDiff'] = 0.5
        params['abvRange'] = np.array([0.0,2.0])

    return params

def get_signal_comb(params,LD,TI,cbf,att,atta,abv):
    '''
    This function allows you to calculate the delta M signal of a single voxel at one time point based on specified parameters.
    The underlying cASL kinetic model is from Buxton 1998. 

    Inputs:
        - params: the dictionary containing parameter specifications
        - LD: the label duration (s), scalar value
        - TI: the inversion time (s), in cASL is PLD+LD, scalar value
        - cbf: ground truth CBF (ml/100g/min), scalar value
        - att: ground truth ATT of tissue (s), scalar value
        - atta: ground truth ATT of macrovascular component (s), scalar value
        - abv: ground truth aBV (%), scalar value

    Outputs:
        signal: delta M signal at one time point, scalar value
    '''
    cbf = cbf/6000
    abv = abv/100
    M_0a = params['M_0a']
    T1_a = params['T1_a']
    alpha = params['labeleff']
    T1_app = 1/(1/params['T1_t']+cbf/params['lambda'])

    # time after label arrives at MV but not EV
    if np.logical_and(TI>=atta,TI<att):
        signal = 2*M_0a*alpha*np.exp(-atta/T1_a)*abv
    # label arrives at EV, but not all MV labels washed out yet, so signal comes from both MV & EV
    elif np.logical_and(att<(atta+LD),np.logical_and(TI>=att,TI<(atta+LD))):
        signal = (2*M_0a*alpha*np.exp(-atta/T1_a)*abv + 
                  2*M_0a*cbf*T1_app*alpha*np.exp(-att/T1_a)*(1-np.exp((att-TI)/T1_app)))
    # label continues to arrive at EV, and all MV labels washed out, signal comes only from EV
    elif np.logical_and(att<(atta+LD),np.logical_and(TI>=(atta+LD),TI<(att+LD))):
        signal = 2*M_0a*cbf*T1_app*alpha*np.exp(-att/T1_a)*(1-np.exp((att-TI)/T1_app))
    # label starts to arrive at EV after all MV labels washed out, no overlap between MV & EV, signal comes only from EV
    elif np.logical_and(att>=(atta+LD),np.logical_and(TI>=att,TI<(att+LD))):
        signal = 2*M_0a*cbf*T1_app*alpha*np.exp(-att/T1_a)*(1-np.exp((att-TI)/T1_app))
    # label starts to wash out the EV
    elif TI>=(att+LD):
        signal = 2*M_0a*cbf*T1_app*alpha*np.exp(-att/T1_a)*np.exp((att+LD-TI)/T1_app)*(1-np.exp(-LD/T1_app))
    else:
        signal = 0
    
    return signal

def get_signal_rand(params,LD,TI,cbf,att,atta,abv):
    '''
    This function allows you to calculate the delta M signal using a map(2D)/volume(3D) of ground truth values 
    at one time point based on specified parameters.The underlying cASL kinetic model is from Buxton 1998. 

    Inputs:
        - params: the dictionary containing parameter specifications
        - LD: the label duration (s), scalar value
        - TI: the inversion time (s), in cASL is PLD+LD, scalar value
        - cbf: ground truth CBF (ml/100g/min), ndarray
        - att: ground truth ATT of tissue (s), ndarray
        - atta: ground truth ATT of macrovascular component (s), ndarray
        - abv: ground truth aBV (%), ndarray

    Outputs:
        signal: delta M signal at one time point, ndarray shape same as cbf
    '''
    cbf = cbf/6000
    abv = abv/100
    M_0a = params['M_0a']
    T1_a = params['T1_a']
    alpha = params['labeleff']
    T1_app = 1/(1/params['T1_t']+cbf/params['lambda'])

    signal = np.zeros(cbf.shape)

    # time after label arrives at MV but not EV
    tRegion1 = np.logical_and(TI>=atta,TI<att)
    signal[tRegion1] = 2*M_0a*alpha*np.exp(-atta[tRegion1]/T1_a)*abv[tRegion1]
    # label arrives at EV, but not all MV labels washed out yet, so signal comes from both MV & EV
    tRegion2 = np.logical_and(att<(atta+LD),np.logical_and(TI>=att,TI<(atta+LD)))
    signal[tRegion2] = (2*M_0a*alpha*np.exp(-atta[tRegion2]/T1_a)*abv[tRegion2] + 
                        2*M_0a*cbf[tRegion2]*T1_app[tRegion2]*alpha*np.exp(-att[tRegion2]/T1_a) * 
                        (1-np.exp((att[tRegion2]-TI)/T1_app[tRegion2])))
    # label continues to arrive at EV, and all MV labels washed out, signal comes only from EV
    tRegion3 = np.logical_and(att<(atta+LD),np.logical_and(TI>=(atta+LD),TI<(att+LD)))
    signal[tRegion3] = (2*M_0a*cbf[tRegion3]*T1_app[tRegion3]*alpha*np.exp(-att[tRegion3]/T1_a) * 
                        (1-np.exp((att[tRegion3]-TI)/T1_app[tRegion3])))
    # label starts to arrive at EV after all MV labels washed out, no overlap between MV & EV, signal comes only from EV
    tRegion4 = np.logical_and(att>=(atta+LD),np.logical_and(TI>=att,TI<(att+LD)))
    signal[tRegion4] = (2*M_0a*cbf[tRegion4]*T1_app[tRegion4]*alpha*np.exp(-att[tRegion4]/T1_a) * 
                        (1-np.exp((att[tRegion4]-TI)/T1_app[tRegion4])))
    # label starts to wash out the EV
    tRegion5 = TI>=(att+LD)
    signal[tRegion5] = (2*M_0a*cbf[tRegion5]*T1_app[tRegion5]*alpha*np.exp(-att[tRegion5]/T1_a) *
                        np.exp((att[tRegion5]+LD-TI)/T1_app[tRegion5])*(1-np.exp(-LD/T1_app[tRegion5])))
    
    return signal

def set_noise(params,SNR):
    '''
    This function allows you to set the noiseSD based on specified parameters and chosen SNR. 
    The SNR is defined as the maximum theoretical ASL signal at a typical condition over noiseSD. 
    The source of the noise is typically considered as coming from the background. 

    Inputs:
        - params: the dictionary containing parameter specifications
        - SNR: signal-to-noise ratio

    Outputs:
        noiseSD: the standard deviation of noise
    '''
    # set a typical protocol (by using the mean values of CVs)
    protocol_standard = set_protocol(params['nDelays'],params['minPLDs'].mean(),params['totalLDs'].mean(),params['delaylins'].mean())
    TIs = protocol_standard['TIs']
    LDs = protocol_standard['LDs']
    
    signal = np.zeros(TIs.shape)
    for i in np.arange(TIs.shape[0]):
        # set typical physiological parameters: CBF = mean of CBF range, ATT = mean of ATT range, no MVC, Buxton model
        signal[i] = get_signal_comb(params,LDs[i],TIs[i],params['cbfRange'].mean(),params['attRange'].mean(),0.,0.)
    
    noiseSD = max(signal)/SNR
    
    return noiseSD

def add_noise(input_signal,noiseSD):
    '''
    This function adds Gaussian white noise to the input signal based on specified noiseSD. 

    Inputs:
        - input_signal: 'noise-free' signal
        - noiseSD: the standard deviation of the white noise

    Outputs:
        noisy_signal: signal with added noise
    '''
    noisy_signal = input_signal+noiseSD*np.random.randn(input_signal.shape[0])
    
    return noisy_signal

def get_signal_decoded(protocol,signal):
    '''
    This function decodes the signal according to the Hadamard matrix of the protocol. For example, if 
    the protocol has a 8x7 Hadamard encoding with 3 repeats, the input signal should be a time series with 
    8x3 time points - nHadAcq1,nHadAcq2,nHadAcq3, the decoded time seried would be 7x3 time points - 
    nHadBlock1,nHadBlock2,nHadBlock3. 

    Inputs:
        - protocol: a dictionary containing protocol specifications, the Hadamard matrix and nRepeat are used
        - signal: acquired signal volumes with Hadamard encoding

    Outputs:
        signal_decoded: Hadamard-decoded signal according to sub-boli of LDs and TIs
    '''
    nRepeat = protocol['repeats']
    nHadAcq = protocol['had'].shape[0]
    nHadBlock = protocol['had'].shape[1]
    
    # signal_decoded is the decoded noisy signal for 1 voxel in n repeats 
    # (e.g. in 7 delay with 3 repeats, a 1x21 array)
    signal_decoded = np.zeros(nRepeat*nHadBlock) 
    
    for m in np.arange(nRepeat):
        # signal_1rpt is the signal block from that repeat only (e.g. 1x8)
        signal_1rpt = signal[m*nHadAcq:(m+1)*nHadAcq] 
        
        # signal_decoded_1rpt the decoded signal block from signal_1rpt (e.g. 1x7)
        signal_decoded_1rpt = np.zeros(nHadBlock)
        
        for i in np.arange(nHadBlock):
            for j in np.arange(nHadAcq):
                if protocol['had'][j,i] == 1:
                    signal_decoded_1rpt[i] = signal_decoded_1rpt[i]+signal_1rpt[j]
            signal_decoded_1rpt[i] = signal_decoded_1rpt[i]/(nHadAcq/2)
        signal_decoded[m*nHadBlock:(m+1)*nHadBlock] = signal_decoded_1rpt
    
    return signal_decoded

def get_image_comb(dim,params,protocol,noiseSD,cbf,att,atta,abv,rawFlag,pureFlag):
    nRepeat = protocol['repeats']
    nHadAcq = protocol['had'].shape[0]
    nHadBlock = protocol['had'].shape[1]
    # calculate the signal segments by each bolus
    signal_segments = np.zeros(nHadBlock)
    for i in np.arange(nHadBlock):
        signal_segments[i] = get_signal_comb(params,protocol['LDs'][i],protocol['TIs'][i],cbf,att,atta,abv)
    # calculate the signal time series in 1 repeat
    signal_1rpt = np.zeros(nHadAcq)
    for i in np.arange(nHadAcq):
        for j in np.arange(nHadBlock):
            if protocol['had'][i,j] == 1:
                signal_1rpt[i] = signal_1rpt[i]+signal_segments[j]
    
    data_raw = np.zeros(np.append(dim,nHadAcq*nRepeat))
    data_decoded = np.zeros(np.append(dim,nHadBlock*nRepeat))
    if pureFlag:
        data_raw_pure = np.zeros(np.append(dim,nHadAcq*nRepeat))
        data_decoded_pure = np.zeros(np.append(dim,nHadBlock*nRepeat))
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            for k in np.arange(dim[2]):
                signal_nrpt = np.tile(signal_1rpt,nRepeat) # # copy by n repeats (--ibf=rpt)
                signal_raw = add_noise(signal_nrpt,noiseSD)
                data_decoded[i,j,k,:] = get_signal_decoded(protocol,signal_raw)
                if pureFlag:
                    data_decoded_pure[i,j,k,:] = get_signal_decoded(protocol,signal_nrpt)
                if rawFlag:
                    data_raw[i,j,k,:] = signal_raw
                    if pureFlag:
                        data_raw_pure[i,j,k,:] = signal_nrpt
    # Write nifti images
    file_prefix = f'cbf_{cbf:d}_att_{att:.1f}_atta_{atta:.1f}_abv_{abv:.1f}'
    nib.save(nib.Nifti1Image(data_decoded,np.eye(4)),file_prefix+'.nii.gz')
    if pureFlag:
        nib.save(nib.Nifti1Image(data_decoded_pure,np.eye(4)),file_prefix+'_pure.nii.gz')
    if rawFlag:
        nib.save(nib.Nifti1Image(data_raw,np.eye(4)),file_prefix+'_raw.nii.gz')
        if pureFlag:
            nib.save(nib.Nifti1Image(data_raw_pure,np.eye(4)),file_prefix+'_raw_pure.nii.gz')
    
    # Append GKM & GKMmvc fabber_asl commands to a text file
    str_mask = '../'+protocol['name']+'_mask.nii.gz'
    str_ld_sep = ''
    str_ti_sep = ''
    str_pld_sep = ''
    for i in np.arange(nHadBlock):
        str_ld_sep = str_ld_sep+f' --tau{i+1:d}={protocol["LDs"][i]:.3f}'
        str_ti_sep = str_ti_sep+f' --ti{i+1:d}={protocol["TIs"][i]:.3f}'
        str_pld_sep = str_pld_sep+f' --pld{i+1:d}={protocol["PLDs"][i]:.3f}'
    str_input = file_prefix+'.nii.gz'
    file_fabber_asl_gkm = open(protocol['name']+'_fabber_asl_gkm.txt','a+')
    comm_fabber_asl_gkm = ('fabber_asl --model=aslrest --method=vb --data='+str_input+
                           ' --output='+file_prefix+'_fabber_asl_gkm --mask='+str_mask+
                           f' --inctiss --infertiss --incbat --inferbat --casl --repeats={nRepeat:d}'+
                           str_ld_sep+str_ti_sep+' --save-mean --save-std --noise=white'+
                           ' --PSP_byname1=delttiss'+f' --PSP_byname1_mean={params["batprior"]:.1f}'+
                           f' --PSP_byname1_prec={params["batsdprior"]:.1f} \n\n')
    file_fabber_asl_gkm.write(comm_fabber_asl_gkm)
    file_fabber_asl_gkm.close()
    file_fabber_asl_gkmmvc = open(protocol['name']+'_fabber_asl_gkm.txt','a+')
    comm_fabber_asl_gkmmvc = ('fabber_asl --model=aslrest --method=vb --data='+str_input+
                              ' --output='+file_prefix+'_fabber_asl_gkmmvc --mask='+str_mask+
                              ' --inctiss --infertiss --incbat --inferbat --incart --inferart'+
                              f' --casl --repeats={nRepeat:d}'+str_ld_sep+str_ti_sep+
                              ' --save-mean --save-std --noise=white'+
                              ' --PSP_byname1=delttiss'+f' --PSP_byname1_mean={params["batprior"]:.1f}'+
                              f' --PSP_byname1_prec={params["batsdprior"]:.1f} \n\n')
    file_fabber_asl_gkmmvc.write(comm_fabber_asl_gkmmvc)
    file_fabber_asl_gkmmvc.close()
    
    # Append GKM & GKMmvc basil commands to a text file
    str_opt = protocol['name']+'_options.txt'
    file_basil_opt = open(protocol['name']+'_options.txt','w+')
    file_basil_opt.write('# basil options file for protocol'+protocol['name']+' \n\n')
    comm_basil_opt = ('--casl'+str_ld_sep+str_pld_sep+f' --repeats={nRepeat:d}'+
                      f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n')
    file_basil_opt.write(comm_basil_opt)
    file_basil_opt.close()
    file_basil_gkm = open(protocol['name']+'_basil_gkm.txt','a+')
    comm_basil_gkm = ('basil -i '+str_input+' -o '+file_prefix+'_basil_gkm -m '+str_mask+
                      ' --optfile '+str_opt+' \n\n')
    file_basil_gkm.write(comm_basil_gkm)
    file_basil_gkm.close()
    file_basil_gkmmvc = open(protocol['name']+'_basil_gkmmvc.txt','a+')
    comm_basil_gkmmvc = ('basil -i '+str_input+' -o '+file_prefix+'_basil_gkmmvc -m '+str_mask+
                         ' --optfile '+str_opt+' --inferart \n\n')
    file_basil_gkmmvc.write(comm_basil_gkmmvc)
    file_basil_gkmmvc.close()
    
    # Append GKM & GKMmvc oxford_asl commands to a text file
    str_ld_all = ''
    str_ti_all = ''
    for i in np.arange(nHadBlock):
        str_ld_all = str_ld_all+f'{protocol["LDs"][i]:.3f}'
        str_ti_all = str_ti_all+f'{protocol["TIs"][i]:.3f}'
        if i != (nHadBlock-1):
            str_ld_all = str_ld_all+','
            str_ti_all = str_ti_all+','
    file_oxford_asl_gkm = open(protocol['name']+'_oxford_asl_gkm.txt','a+')
    comm_oxford_asl_gkm = ('oxford_asl -i '+str_input+' -o '+file_prefix+'_oxford_asl_gkm -m '+str_mask+
                           ' --casl --bolus='+str_ld_all+' --tis='+str_ti_all+f' --rpts={nRepeat:d}'+
                           ' --spatial=off --artoff'+
                           f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n\n')
    file_oxford_asl_gkm.write(comm_oxford_asl_gkm)
    file_oxford_asl_gkm.close()
    file_oxford_asl_gkmmvc = open(protocol['name']+'_oxford_asl_gkmmvc.txt','a+')
    comm_oxford_asl_gkmmvc = ('oxford_asl -i '+str_input+' -o '+file_prefix+'_oxford_asl_gkmmvc -m '+str_mask+
                              ' --casl --bolus='+str_ld_all+' --tis='+str_ti_all+f' --rpts={nRepeat:d}'+
                              ' --spatial=off'+
                              f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n\n')
    file_oxford_asl_gkmmvc.write(comm_oxford_asl_gkmmvc)
    file_oxford_asl_gkmmvc.close()
    
    return None

def get_image_rand(dim,params,protocol,noiseSD,cbfRange,attRange,abvRange,rawFlag,mvcFlag,pureFlag):
    nRepeat = protocol['repeats']
    nHadAcq = protocol['had'].shape[0]
    nHadBlock = protocol['had'].shape[1]
    
    nomvc_gt_cbf = cbfRange[0]+(cbfRange[1]-cbfRange[0])*np.random.rand(dim[0],dim[1],dim[2])
    nomvc_gt_att = attRange[0]+(attRange[1]-attRange[0])*np.random.rand(dim[0],dim[1],dim[2])
    nomvc_gt_atta = np.zeros(dim)
    nomvc_gt_abv = np.zeros(dim)
    if mvcFlag:
        mvc_gt_cbf = cbfRange[0]+(cbfRange[1]-cbfRange[0])*np.random.rand(dim[0],dim[1],dim[2])
        mvc_gt_att = attRange[0]+(attRange[1]-attRange[0])*np.random.rand(dim[0],dim[1],dim[2])
        mvc_gt_atta = mvc_gt_att - params['attDiff']
        mvc_gt_abv = abvRange[0]+(abvRange[1]-abvRange[0])*np.random.rand(dim[0],dim[1],dim[2])
    
    # calculate the signal segments by each bolus
    nomvc_signal_segments = np.zeros([nHadBlock,dim[0],dim[1],dim[2]])
    if mvcFlag:
        mvc_signal_segments = np.zeros([nHadBlock,dim[0],dim[1],dim[2]])
    for i in np.arange(nHadBlock):
        nomvc_signal_segments[i] = get_signal_rand(params,protocol['LDs'][i],protocol['TIs'][i],
                                                           nomvc_gt_cbf,nomvc_gt_att,nomvc_gt_atta,nomvc_gt_abv)
        if mvcFlag:
            mvc_signal_segments[i] = get_signal_rand(params,protocol['LDs'][i],protocol['TIs'][i],
                                                             mvc_gt_cbf,mvc_gt_att,mvc_gt_atta,mvc_gt_abv)
    # calculate the signal volumes in 1 repeat
    nomvc_signal_1rpt = np.zeros([nHadAcq,dim[0],dim[1],dim[2]])
    if mvcFlag:
        mvc_signal_1rpt = np.zeros([nHadAcq,dim[0],dim[1],dim[2]])
    for i in np.arange(nHadAcq):
        for j in np.arange(nHadBlock):
            if protocol['had'][i,j] == 1:
                nomvc_signal_1rpt[i] = nomvc_signal_1rpt[i]+nomvc_signal_segments[j]
        if mvcFlag:
            for j in np.arange(nHadBlock):
                if protocol['had'][i,j] == 1:
                    mvc_signal_1rpt[i] = mvc_signal_1rpt[i]+mvc_signal_segments[j]
    
    nomvc_data_raw = np.zeros(np.append(dim,nHadAcq*nRepeat))
    nomvc_data_decoded = np.zeros(np.append(dim,nHadBlock*nRepeat))
    if pureFlag:
        nomvc_data_raw_pure = np.zeros(np.append(dim,nHadAcq*nRepeat))
        nomvc_data_decoded_pure = np.zeros(np.append(dim,nHadBlock*nRepeat))
    if mvcFlag:
        mvc_data_raw = np.zeros(np.append(dim,nHadAcq*nRepeat))
        mvc_data_decoded = np.zeros(np.append(dim,nHadBlock*nRepeat))
        if pureFlag:
            mvc_data_raw_pure = np.zeros(np.append(dim,nHadAcq*nRepeat))
            mvc_data_decoded_pure = np.zeros(np.append(dim,nHadBlock*nRepeat))
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            for k in np.arange(dim[2]):
                # nomvc_signal_nrpt is the undecoded pure signal for 1 voxel in n repeats 
                # (e.g. in 7 delay with 3 repeats, a 1x24 array)
                nomvc_signal_nrpt = np.tile(nomvc_signal_1rpt[:,i,j,k],nRepeat) # copy by n repeats (--ibf=rpt)
                # nomvc_signal_raw is the undecoded noisy signal for 1 voxel in n repeats
                nomvc_signal_raw = add_noise(nomvc_signal_nrpt,noiseSD)
                nomvc_data_decoded[i,j,k,:] = get_signal_decoded(protocol,nomvc_signal_raw)
                if pureFlag:
                    nomvc_data_decoded_pure[i,j,k,:] = get_signal_decoded(protocol,nomvc_signal_nrpt)
                if rawFlag:
                    nomvc_data_raw[i,j,k,:] = nomvc_signal_raw
                    if pureFlag:
                        nomvc_data_raw_pure[i,j,k,:] = nomvc_signal_nrpt
                
                if mvcFlag:
                    mvc_signal_nrpt = np.tile(mvc_signal_1rpt[:,i,j,k],nRepeat) # copy by n repeats (--ibf=rpt)
                    # mvc_signal_raw is the undecoded noisy signal for 1 voxel in n repeats
                    mvc_signal_raw = add_noise(mvc_signal_nrpt,noiseSD)
                    mvc_data_decoded[i,j,k,:] = get_signal_decoded(protocol,mvc_signal_raw)
                    if pureFlag:
                        mvc_data_decoded_pure[i,j,k,:] = get_signal_decoded(protocol,mvc_signal_nrpt)
                    if rawFlag:
                        mvc_data_raw[i,j,k,:] = mvc_signal_raw
                        if pureFlag:
                            mvc_data_raw_pure[i,j,k,:] = mvc_signal_nrpt
    # Write nifti images
    nib.save(nib.Nifti1Image(nomvc_data_decoded,np.eye(4)),'nomvc_asl.nii.gz')
    if pureFlag:
        nib.save(nib.Nifti1Image(nomvc_data_decoded_pure,np.eye(4)),'nomvc_asl_pure.nii.gz')
    nib.save(nib.Nifti1Image(nomvc_gt_cbf,np.eye(4)),'nomvc_gt_cbf.nii.gz')
    nib.save(nib.Nifti1Image(nomvc_gt_att,np.eye(4)),'nomvc_gt_att.nii.gz')
    if rawFlag:
        nib.save(nib.Nifti1Image(nomvc_data_raw,np.eye(4)),'nomvc_asl_raw.nii.gz')
        if pureFlag:
            nib.save(nib.Nifti1Image(nomvc_data_raw_pure,np.eye(4)),'nomvc_asl_raw_pure.nii.gz')
    if mvcFlag:
        nib.save(nib.Nifti1Image(mvc_data_decoded,np.eye(4)),'mvc_asl.nii.gz')
        if pureFlag:
            nib.save(nib.Nifti1Image(mvc_data_decoded_pure,np.eye(4)),'mvc_asl_pure.nii.gz')
        nib.save(nib.Nifti1Image(mvc_gt_cbf,np.eye(4)),'mvc_gt_cbf.nii.gz')
        nib.save(nib.Nifti1Image(mvc_gt_att,np.eye(4)),'mvc_gt_att.nii.gz')
        nib.save(nib.Nifti1Image(mvc_gt_atta,np.eye(4)),'mvc_gt_atta.nii.gz')
        nib.save(nib.Nifti1Image(mvc_gt_abv,np.eye(4)),'mvc_gt_abv.nii.gz')
        if rawFlag:
            nib.save(nib.Nifti1Image(mvc_data_raw,np.eye(4)),'mvc_asl_raw.nii.gz')
            if pureFlag:
                nib.save(nib.Nifti1Image(mvc_data_raw_pure,np.eye(4)),'mvc_asl_raw_pure.nii.gz')
            
    # Append GKM & GKMmvc fabber_asl commands to a text file
    str_mask = '../'+protocol['name']+'_mask.nii.gz'
    str_ld_sep = ''
    str_ti_sep = ''
    str_pld_sep = ''
    for i in np.arange(nHadBlock):
        str_ld_sep = str_ld_sep+f' --tau{i+1:d}={protocol["LDs"][i]:.3f}'
        str_ti_sep = str_ti_sep+f' --ti{i+1:d}={protocol["TIs"][i]:.3f}'
        str_pld_sep = str_pld_sep+f' --pld{i+1:d}={protocol["PLDs"][i]:.3f}'
    file_fabber_asl_gkm = open(protocol['name']+'_fabber_asl_gkm.txt','a+')
    comm_fabber_asl_gkm = ('fabber_asl --model=aslrest --method=vb --data=nomvc_asl.nii.gz'+
                           ' --output=nomvc_asl_fabber_asl_gkm --mask='+str_mask+
                           f' --inctiss --infertiss --incbat --inferbat --casl --repeats={nRepeat:d}'+
                           str_ld_sep+str_ti_sep+' --save-mean --save-std --noise=white'+
                           ' --PSP_byname1=delttiss'+f' --PSP_byname1_mean={params["batprior"]:.1f}'+
                           f' --PSP_byname1_prec={params["batsdprior"]:.1f} \n\n')
    file_fabber_asl_gkm.write(comm_fabber_asl_gkm)
    if mvcFlag:
        mvc_comm_fabber_asl_gkm = ('fabber_asl --model=aslrest --method=vb --data=mvc_asl.nii.gz'+
                                   ' --output=mvc_asl_fabber_asl_gkm --mask='+str_mask+
                                   f' --inctiss --infertiss --incbat --inferbat --casl --repeats={nRepeat:d}'+
                                   str_ld_sep+str_ti_sep+' --save-mean --save-std --noise=white'+
                                   ' --PSP_byname1=delttiss'+f' --PSP_byname1_mean={params["batprior"]:.1f}'+
                                   f' --PSP_byname1_prec={params["batsdprior"]:.1f} \n\n')
        file_fabber_asl_gkm.write(mvc_comm_fabber_asl_gkm)
        file_fabber_asl_gkmmvc = open(protocol['name']+'_fabber_asl_gkmmvc.txt','a+')
        mvc_comm_fabber_asl_gkmmvc = ('fabber_asl --model=aslrest --method=vb --data=mvc_asl.nii.gz'+
                                      ' --output=mvc_asl_fabber_asl_gkm --mask='+str_mask+
                                      ' --inctiss --infertiss --incbat --inferbat --incart --inferart'+
                                      f' --casl --repeats={nRepeat:d}'+str_ld_sep+str_ti_sep+
                                      ' --save-mean --save-std --noise=white'+
                                      ' --PSP_byname1=delttiss'+f' --PSP_byname1_mean={params["batprior"]:.1f}'+
                                      f' --PSP_byname1_prec={params["batsdprior"]:.1f} \n\n')
        file_fabber_asl_gkmmvc.write(mvc_comm_fabber_asl_gkmmvc)
        file_fabber_asl_gkmmvc.close()
    file_fabber_asl_gkm.close()
    
    # Append GKM & GKMmvc basil commands to a text file
    str_opt = protocol['name']+'_options.txt'
    file_basil_opt = open(protocol['name']+'_options.txt','a+')
    file_basil_opt.write('# basil options file for protocol'+protocol['name']+' \n\n')
    comm_basil_opt = ('--casl'+str_ld_sep+str_pld_sep+f' --repeats={nRepeat:d}'+
                      f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n')
    file_basil_opt.write(comm_basil_opt)
    file_basil_opt.close()
    file_basil_gkm = open(protocol['name']+'_basil_gkm.txt','a+')
    comm_basil_gkm = ('basil -i nomvc_asl.nii.gz -o nomvc_asl_basil_gkm -m '+str_mask+
                      ' --optfile '+str_opt+' \n\n')
    file_basil_gkm.write(comm_basil_gkm)
    if mvcFlag:
        mvc_comm_basil_gkm = ('basil -i mvc_asl.nii.gz -o mvc_asl_basil_gkm -m '+str_mask+
                              ' --optfile '+str_opt+' \n\n')
        file_basil_gkm.write(mvc_comm_basil_gkm)
        file_basil_gkmmvc = open(protocol['name']+'_basil_gkmmvc.txt','a+')
        mvc_comm_basil_gkmmvc = ('basil -i mvc_asl.nii.gz -o mvc_asl_basil_gkmmvc -m '+str_mask+
                                 ' --optfile '+str_opt+' --inferart \n\n')
        file_basil_gkmmvc.write(mvc_comm_basil_gkmmvc)
        file_basil_gkmmvc.close()
    file_basil_gkm.close()
    
    # Append GKM & GKMmvc oxford_asl commands to a text file
    str_ld_all = ''
    str_ti_all = ''
    for i in np.arange(nHadBlock):
        str_ld_all = str_ld_all+f'{protocol["LDs"][i]:.3f}'
        str_ti_all = str_ti_all+f'{protocol["TIs"][i]:.3f}'
        if i != (nHadBlock-1):
            str_ld_all = str_ld_all+','
            str_ti_all = str_ti_all+','
    file_oxford_asl_gkm = open(protocol['name']+'_oxford_asl_gkm.txt','a+')
    comm_oxford_asl_gkm = ('oxford_asl -i nomvc_asl.nii.gz -o nomvc_asl_oxford_asl_gkm -m '+str_mask+
                           ' --casl --bolus='+str_ld_all+' --tis='+str_ti_all+f' --rpts={nRepeat:d}'+
                           ' --spatial=off --artoff'+
                           f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n\n')
    file_oxford_asl_gkm.write(comm_oxford_asl_gkm)
    if mvcFlag:
        mvc_comm_oxford_asl_gkm = ('oxford_asl -i mvc_asl.nii.gz -o mvc_asl_oxford_asl_gkm -m '+str_mask+
                                   ' --casl --bolus='+str_ld_all+' --tis='+str_ti_all+f' --rpts={nRepeat:d}'+
                                   ' --spatial=off --artoff'+
                                   f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n\n')
        file_oxford_asl_gkm.write(mvc_comm_oxford_asl_gkm)
        file_oxford_asl_gkmmvc = open(protocol['name']+'_oxford_asl_gkmmvc.txt','a+')
        mvc_comm_oxford_asl_gkmmvc = ('oxford_asl -i mvc_asl.nii.gz -o mvc_asl_oxford_asl_gkmmvc -m '+str_mask+
                                      ' --casl --bolus='+str_ld_all+' --tis='+str_ti_all+f' --rpts={nRepeat:d}'+
                                      ' --spatial=off'+
                                      f' --bat={params["batprior"]:.1f} --batsd={params["batsdprior"]:.1f} \n\n')
        file_oxford_asl_gkmmvc.write(mvc_comm_oxford_asl_gkmmvc)
        file_oxford_asl_gkmmvc.close()
    file_oxford_asl_gkm.close()

    return None

def run_simulation(rootdir,params,specification):

    params['noiseSD'] = set_noise(params,params['SNR'])

    date = datetime.today().strftime('%Y%m%d')
    dir_output = rootdir+'sim_'+date+'_'+specification+'/'
    os.mkdir(dir_output)
    os.chdir(dir_output)

    if params['mode'] == 'comb':
        nProtocol = 0; NProtocol = len(params['minPLDs'])*len(params['totalLDs'])*len(params['delaylins'])
        nProcess = 0; NProcess = 0
        for minPLD in params['minPLDs']:
            for totalLD in params['totalLDs']:
                for delaylin in params['delaylins']:
                    for cbf in params['cbfRange']:
                        for att in params['attRange']:
                            attaValues = np.arange(params['atta_start'],att,params['atta_step'])
                            for atta in attaValues:
                                for abv in params['abvRange']:
                                    NProcess += 1
        for minPLD in params['minPLDs']:
            for totalLD in params['totalLDs']:
                for delaylin in params['delaylins']:
                    protocol = set_protocol(params['nDelays'],minPLD,totalLD,delaylin)
                    nProtocol += 1
                    dir_protocol = dir_output+protocol['name']+'/'
                    os.mkdir(dir_protocol)
                    os.chdir(dir_protocol)
                    # Save mask image
                    nib.save(nib.Nifti1Image(np.ones(params['dim']),np.eye(4)),protocol['name']+'_mask.nii.gz')
                    # Simulate images
                    dir_data = dir_protocol+'data/'
                    os.mkdir(dir_data)
                    os.chdir(dir_data)
                    for cbf in params['cbfRange']:
                        for att in params['attRange']:
                            attaValues = np.arange(params['atta_start'],att,params['atta_step'])
                            for atta in attaValues:
                                for abv in params['abvRange']:
                                    clear_output(wait=True)
                                    nProcess += 1
                                    print('Simulating protocol',protocol['name'],'(',nProtocol,'of',NProtocol,')',
                                          f'cbf={cbf:d} att={att:.1f} atta={atta:.1f} abv={abv:.1f}','...')
                                    print(f'************  Simulation Process: {nProcess/NProcess*100:.2f}%  ************')
                                    get_image_comb(params['dim'],params,protocol,params['noiseSD'],
                                                              cbf,att,atta,abv,
                                                              rawFlag=False,pureFlag=False)
                    os.chdir(dir_protocol)
                    # Combining all the commands
                    file_protocolExe = open(protocol['name']+'_exe.txt','a+')
                    file_protocolExe.write('# BASH fabber execution file \r\n')
                    file_protocolExe.write('cd ./data/ \r\n')
                    file_protocolExe.write('sh -e '+protocol['name']+'_'+params['fsltool']+'_gkm.txt \r\n')
                    file_protocolExe.write('sh -e '+protocol['name']+'_'+params['fsltool']+'_gkmmvc.txt \r\n')
                    file_protocolExe.write('cd ../ \r\n')
                    file_protocolExe.close()
                    
                    sio.savemat('simInfo_protocol.mat',protocol)
                    os.chdir(dir_output)
                    
                    file_simulationExe = open(specification+'_exe.txt','a+')
                    file_simulationExe.write('cd ./'+protocol['name']+' \r\n')
                    file_simulationExe.write('sh -e '+protocol['name']+'_exe.txt \r\n')
                    file_simulationExe.write('cd ../ \r\n')
                    file_simulationExe.close()
                    
        sio.savemat('simInfo_parameters.mat',params)
        os.chdir(rootdir)
        print('Simulation complete!')

    if params['mode'] == 'rand':
        nProtocol = 0; NProtocol = len(params['minPLDs'])*len(params['totalLDs'])*len(params['delaylins'])
        for minPLD in params['minPLDs']:
            for totalLD in params['totalLDs']:
                for delaylin in params['delaylins']:
                    protocol = set_protocol(params['nDelays'],minPLD,totalLD,delaylin)
                    clear_output(wait=True)
                    nProtocol += 1
                    print('Simulating protocol',protocol['name'],'(',nProtocol,'of',NProtocol,')','...')
                    print(f'************  Simulation Process: {nProtocol/NProtocol*100:.2f}%  ************')
                    dir_protocol = dir_output+protocol['name']+'/'
                    os.mkdir(dir_protocol)
                    os.chdir(dir_protocol)
                    # Save mask image
                    nib.save(nib.Nifti1Image(np.ones(params['dim']),np.eye(4)),protocol['name']+'_mask.nii.gz')
                    # Simulate images
                    dir_data = dir_protocol+'data/'
                    os.mkdir(dir_data)
                    os.chdir(dir_data)
                    get_image_rand(params['dim'],params,protocol,params['noiseSD'],
                                              params['cbfRange'],params['attRange'],params['abvRange'],
                                              rawFlag=False,mvcFlag=True,pureFlag=False)
                    os.chdir(dir_protocol)
                    
                    # Combining all the commands
                    file_protocolExe = open(protocol['name']+'_exe.txt','a+')
                    file_protocolExe.write('# BASH fabber execution file \r\n')
                    file_protocolExe.write('cd ./data/ \r\n')
                    file_protocolExe.write('sh -e '+protocol['name']+'_'+params['fsltool']+'_gkm.txt \r\n')
                    file_protocolExe.write('sh -e '+protocol['name']+'_'+params['fsltool']+'_gkmmvc.txt \r\n')
                    file_protocolExe.write('cd ../ \r\n')
                    file_protocolExe.close()
                    
                    sio.savemat('simInfo_protocol.mat',protocol)
                    os.chdir(dir_output)
                    
                    file_simulationExe = open(specification+'_exe.txt','a+')
                    file_simulationExe.write('cd ./'+protocol['name']+' \r\n')
                    file_simulationExe.write('sh -e '+protocol['name']+'_exe.txt \r\n')
                    file_simulationExe.write('cd ../ \r\n')
                    file_simulationExe.close()
                    
        sio.savemat('simInfo_parameters.mat',params)
        os.chdir(rootdir)
        print('Simulation complete!')

    return None

def get_protocol_properties(params):

    noiseSD = set_noise(params,SNR=params['SNR'])
    dim = params['nSamp']
    minPLDs = params['minPLDs']
    totalLDs = params['totalLDs']
    delaylins = params['delaylins']

    nomvc_gt_cbf = np.tile(np.linspace(params['cbfRange'][0],params['cbfRange'][0],dim),(dim,1))
    nomvc_gt_att = np.tile(np.reshape(np.linspace(params['attRange'][0],params['attRange'][1],dim),[dim,1]),(1,dim))
    nomvc_gt_atta = np.zeros(nomvc_gt_cbf.shape)
    nomvc_gt_abv = np.zeros(nomvc_gt_cbf.shape)

    nomvc_props = {}
    nomvc_props['scantime_1rpt'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['scantime'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['nRepeatIn1Block'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['idletime'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['nVolume'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['signalSqrtSum'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['wholeSNR'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    nomvc_props['tSNR'] = np.full([len(minPLDs),len(totalLDs),len(delaylins)],np.nan)
    
    nProtocol = 0; NProtocol = len(minPLDs)*len(totalLDs)*len(delaylins)

    for minPLD in enumerate(minPLDs):
        for totalLD in enumerate(totalLDs):
            for delaylin in enumerate(delaylins):
                protocol = set_protocol(params['nDelays'],minPLD[1],totalLD[1],delaylin[1])
                clear_output(wait=True)
                nProtocol += 1
                print('Calculating properties of protocol',protocol['name'],'(',nProtocol,'of',NProtocol,')','...')
                print(f'************  Calculation Process: {nProtocol/NProtocol*100:.2f}%  ************')
                nRepeat = protocol['repeats']
                nHadBlock = protocol['had'].shape[1]
                signalAcq = np.zeros(nRepeat*nHadBlock)
                for m in np.arange(nRepeat):
                    for i in np.arange(nHadBlock):
                        index = m*nHadBlock+i
                        signalAcq[index] = get_signal_rand(params,protocol['LDs'][i],protocol['TIs'][i],
                                                                   nomvc_gt_cbf,nomvc_gt_att,nomvc_gt_atta,nomvc_gt_abv).mean()
                nomvc_props['scantime_1rpt'][minPLD[0]][totalLD[0]][delaylin[0]] = protocol['scantime_1rpt']
                nomvc_props['scantime'][minPLD[0]][totalLD[0]][delaylin[0]] = protocol['scantime']
                nomvc_props['nRepeatIn1Block'][minPLD[0]][totalLD[0]][delaylin[0]] = np.floor(params['blocktime']/protocol['scantime_1rpt'])
                nomvc_props['idletime'][minPLD[0]][totalLD[0]][delaylin[0]] = params['scantimelim'] - protocol['scantime']
                nomvc_props['nVolume'][minPLD[0]][totalLD[0]][delaylin[0]] = len(signalAcq)
                nomvc_props['signalSqrtSum'][minPLD[0]][totalLD[0]][delaylin[0]] = np.sqrt(np.sum(signalAcq))
                nomvc_props['wholeSNR'][minPLD[0]][totalLD[0]][delaylin[0]] = np.sqrt(np.sum(signalAcq))/noiseSD
                nomvc_props['tSNR'][minPLD[0]][totalLD[0]][delaylin[0]] = np.sqrt(np.sum(signalAcq))/noiseSD/protocol['scantime']

    return nomvc_props

def visualise_properties(params,nomvc_props,property):
    # Print the maximum property index
    maxindex = np.unravel_index(np.argmax(nomvc_props[property],axis=None),nomvc_props[property].shape)
    print('The maximum property',property,f'is {nomvc_props[property][maxindex]:.2f}',
          f', and appears at minPLD = {params["minPLDs"][maxindex[0]]:.1f}',
          f', totalLD = {params["totalLDs"][maxindex[1]]:.1f}',
          f', delaylin = {params["delaylins"][maxindex[2]]:.1f}')
    # Print the minumum property index
    minindex = np.unravel_index(np.argmin(nomvc_props[property],axis=None),nomvc_props[property].shape)
    print('The minimum property',property,f'is {nomvc_props[property][minindex]:.2f}',
          f', and appears at minPLD = {params["minPLDs"][minindex[0]]:.1f}',
          f', totalLD = {params["totalLDs"][minindex[1]]:.1f}',
          f', delaylin = {params["delaylins"][minindex[2]]:.1f}')
    # Visualise property using 3D Graph plot
    X,Y,Z = np.mgrid[params['minPLDs'][0]:params['minPLDs'][-1]:len(params['minPLDs'])*1j,
                     params['totalLDs'][0]:params['totalLDs'][-1]:len(params['totalLDs'])*1j,
                     params['delaylins'][0]:params['delaylins'][-1]:len(params['delaylins'])*1j]
    fig = go.Figure(data=go.Volume(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=nomvc_props[property].flatten(),
                                   opacity=0.1,colorscale='Viridis',surface_count=25))
    fig.update_layout(scene = dict(xaxis_title='CV4 = minPLD (s)',
                                   yaxis_title='CV5 = totalLD (s)',
                                   zaxis_title='CV7 = delaylin'),width=700,margin=dict(r=20,b=10,l=10,t=10))
    return fig

def visualise_3d(xarray,yarray,zarray,fxyz,titles_list,findmax=True,findmin=True,findlb=[float('nan'),float('nan'),float('nan')],findub=[float('nan'),float('nan'),float('nan')]):
    if np.isnan(findlb).all():
        findlb = [xarray[0],yarray[0],zarray[0]]
    if np.isnan(findub).all():
        findub = [xarray[-1],yarray[-1],zarray[-1]]
    x_lb = np.where(np.abs(xarray-findlb[0])<0.001)[0][0]; x_ub = np.where(np.abs(xarray-findub[0])<0.001)[0][0]
    y_lb = np.where(np.abs(yarray-findlb[1])<0.001)[0][0]; y_ub = np.where(np.abs(yarray-findub[1])<0.001)[0][0]
    z_lb = np.where(np.abs(zarray-findlb[2])<0.001)[0][0]; z_ub = np.where(np.abs(zarray-findub[2])<0.001)[0][0]
    xarray_sub = xarray[x_lb:x_ub+1]
    yarray_sub = yarray[y_lb:y_ub+1]
    zarray_sub = zarray[z_lb:z_ub+1]
    fxyz_sub = fxyz[x_lb:x_ub,y_lb:y_ub,z_lb:z_ub]
    if np.logical_or(findmax,findmin):
        print('The search boundary is',f'{titles_list[0]}: {xarray_sub[0]:.1f} to {xarray_sub[-1]:.1f},',
              f'{titles_list[1]}: {yarray_sub[0]:.1f} to {yarray_sub[-1]:.1f},',f'{titles_list[2]}: {zarray_sub[0]:.1f} to {zarray_sub[-1]:.1f}')
    if findmax:
        maxindex = np.unravel_index(np.nanargmax(fxyz_sub,axis=None),fxyz_sub.shape)
        print(f'The max value is {fxyz_sub[maxindex]:.2f},',f'appearing at {titles_list[0]} = {xarray_sub[maxindex[0]]:.1f},',
              f'{titles_list[1]} = {yarray_sub[maxindex[1]]:.1f},',f'{titles_list[2]} = {zarray_sub[maxindex[2]]:.1f}')
    if findmin:
        minindex = np.unravel_index(np.nanargmin(fxyz_sub,axis=None),fxyz_sub.shape)
        print(f'The min value is {fxyz_sub[minindex]:.2f},',f'appearing at {titles_list[0]} = {xarray_sub[minindex[0]]:.1f},',
              f'{titles_list[1]} = {yarray_sub[minindex[1]]:.1f},',f'{titles_list[2]} = {zarray_sub[minindex[2]]:.1f}')
    # Visualise fxyz using 3D Graph plot
    X,Y,Z = np.mgrid[xarray[0]:xarray[-1]:len(xarray)*1j,yarray[0]:yarray[-1]:len(yarray)*1j,zarray[0]:zarray[-1]:len(zarray)*1j]
    fig = go.Figure(data=go.Volume(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=fxyz.flatten(),opacity=0.1,colorscale='Viridis',surface_count=25))
    fig.update_layout(scene=dict(xaxis_title=titles_list[0],yaxis_title=titles_list[1],zaxis_title=titles_list[2]),width=700,margin=dict(r=20,b=10,l=10,t=10))

    return fig

def get_estimation_oxford_asl(estdir,params):
    
    results = {}
    cbf_calib = 6000
    abv_calib = 100
    minPLDs = params['minPLDs']
    totalLDs = params['totalLDs']
    delaylins = params['delaylins']
    NProcess = 0
    protocolList = []
    for minPLD in minPLDs:
        for totalLD in totalLDs:
            for delaylin in delaylins:
                protocolList = np.append(protocolList,f'GEeASL{params["nDelays"]:d}-CV4-{minPLD:.1f}-CV5-{totalLD:.1f}-CV7-{delaylin:.1f}')
                if params['mode'] == 'comb':
                    for cbf in params['cbfRange']:
                        for att in params['attRange']:
                            attaValues = np.arange(params['atta_start'],att+params['atta_step'],params['atta_step'])
                            for atta in attaValues:
                                for abv in params['abvRange']:
                                    NProcess += 1
                if params['mode'] == 'rand':
                    NProcess += 1
    nProtocol = 0
    nProcess = 0
    for protocol in protocolList:
        nProtocol += 1
        if params['mode'] == 'comb':
            for cbf in params['cbfRange']:
                for att in params['attRange']:
                    attaValues = np.arange(params['atta_start'],att+params['atta_step'],params['atta_step'])
                    for atta in attaValues:
                        for abv in params['abvRange']:
                            nProcess += 1
                            folder = f'cbf_{cbf:d}_att_{att:.1f}_atta_{atta:.1f}_abv_{abv:.1f}_gkm/native_space/'
                            filedir = estdir+protocol+'/data/'+folder
                            keyprefix = protocol+f'_cbf_{cbf:d}_att_{att:.1f}_atta_{atta:.1f}_abv_{abv:.1f}_'
                            results[keyprefix+'gkm_cbfmean'] = nib.load(filedir+'perfusion.nii.gz').get_fdata().flatten()*cbf_calib
                            results[keyprefix+'gkm_cbfvar'] = nib.load(filedir+'perfusion_var.nii.gz').get_fdata().flatten()*cbf_calib*cbf_calib
                            results[keyprefix+'gkm_cbfsd'] = np.sqrt(results[keyprefix+'gkm_cbfvar'])
                            results[keyprefix+'gkm_cbferr'] = (results[keyprefix+'gkm_cbfmean']-cbf)/cbf*100
                            results[keyprefix+'gkm_cbfabserr'] = np.abs(results[keyprefix+'gkm_cbfmean']-cbf)/cbf*100

                            results[keyprefix+'gkm_attmean'] = nib.load(filedir+'arrival.nii.gz').get_fdata().flatten()
                            results[keyprefix+'gkm_attvar'] = nib.load(filedir+'arrival_var.nii.gz').get_fdata().flatten()
                            results[keyprefix+'gkm_attsd'] = np.sqrt(results[keyprefix+'gkm_attvar'])
                            results[keyprefix+'gkm_atterr'] = (results[keyprefix+'gkm_attmean']-att)/att*100
                            results[keyprefix+'gkm_attabserr'] = np.abs(results[keyprefix+'gkm_attmean']-att)/att*100

                            folder = f'cbf_{cbf:d}_att_{att:.1f}_atta_{atta:.1f}_abv_{abv:.1f}_gkmmvc/native_space/'
                            filedir = estdir+protocol+'/data/'+folder
                            results[keyprefix+'gkmmvc_cbfmean'] = nib.load(filedir+'perfusion.nii.gz').get_fdata().flatten()*cbf_calib
                            results[keyprefix+'gkmmvc_cbfvar'] = nib.load(filedir+'perfusion_var.nii.gz').get_fdata().flatten()*cbf_calib*cbf_calib
                            results[keyprefix+'gkmmvc_cbfsd'] = np.sqrt(results[keyprefix+'gkmmvc_cbfvar'])
                            results[keyprefix+'gkmmvc_cbferr'] = (results[keyprefix+'gkmmvc_cbfmean']-cbf)/cbf*100
                            results[keyprefix+'gkmmvc_cbfabserr'] = np.abs(results[keyprefix+'gkmmvc_cbfmean']-cbf)/cbf*100

                            results[keyprefix+'gkmmvc_attmean'] = nib.load(filedir+'arrival.nii.gz').get_fdata().flatten()
                            results[keyprefix+'gkmmvc_attvar'] = nib.load(filedir+'arrival_var.nii.gz').get_fdata().flatten()
                            results[keyprefix+'gkmmvc_attsd'] = np.sqrt(results[keyprefix+'gkmmvc_attvar'])
                            results[keyprefix+'gkmmvc_atterr'] = (results[keyprefix+'gkmmvc_attmean']-att)/att*100
                            results[keyprefix+'gkmmvc_attabserr'] = np.abs(results[keyprefix+'gkmmvc_attmean']-att)/att*100

                            results[keyprefix+'gkmmvc_abv'] = nib.load(filedir+'aCBV.nii.gz').get_fdata().flatten()*abv_calib
                            if abv<0.01:
                                results[keyprefix+'gkmmvc_abverr'] = np.full(len(results[keyprefix+'gkmmvc_abv']),np.nan)
                                results[keyprefix+'gkmmvc_abvabserr'] = np.full(len(results[keyprefix+'gkmmvc_abv']),np.nan)
                            else:
                                results[keyprefix+'gkmmvc_abverr'] = (results[keyprefix+'gkmmvc_abv']-abv)/abv*100
                                results[keyprefix+'gkmmvc_abvabserr'] = np.abs(results[keyprefix+'gkmmvc_abv']-abv)/abv*100
                            clear_output(wait=True)
                            print('Collecting estimation results from protocol',protocol,'(',nProtocol,'of',len(protocolList),')',
                                f'cbf={cbf:d} att={att:.1f} atta={atta:.1f} abv={abv:.1f}','...')
                            print(f'***************  Process: {nProcess/NProcess*100:.2f}%  ***************')
        if params['mode'] == 'rand':
            nProcess += 1
            datadir = estdir+protocol+'/data/'
            results[protocol+'_nomvc_asl_gkm_cbfmean'] = nib.load(datadir+'nomvc_asl_oxford_asl_gkm/native_space/perfusion.nii.gz').get_fdata().flatten()*cbf_calib
            results[protocol+'_nomvc_asl_gkm_cbfvar'] = nib.load(datadir+'nomvc_asl_oxford_asl_gkm/native_space/perfusion_var.nii.gz').get_fdata().flatten()*cbf_calib*cbf_calib
            results[protocol+'_nomvc_asl_gkm_cbfsd'] = np.sqrt(results[protocol+'_nomvc_asl_gkm_cbfvar'])
            results[protocol+'_nomvc_asl_gkm_cbfgt'] = nib.load(datadir+'nomvc_gt_cbf.nii.gz').get_fdata().flatten()
            results[protocol+'_nomvc_asl_gkm_cbferr'] = (results[protocol+'_nomvc_asl_gkm_cbfmean']-results[protocol+'_nomvc_asl_gkm_cbfgt'])/results[protocol+'_nomvc_asl_gkm_cbfgt']*100
            results[protocol+'_nomvc_asl_gkm_cbfabserr'] = np.abs(results[protocol+'_nomvc_asl_gkm_cbfmean']-results[protocol+'_nomvc_asl_gkm_cbfgt'])/results[protocol+'_nomvc_asl_gkm_cbfgt']*100

            results[protocol+'_nomvc_asl_gkm_attmean'] = nib.load(datadir+'nomvc_asl_oxford_asl_gkm/native_space/arrival.nii.gz').get_fdata().flatten()
            results[protocol+'_nomvc_asl_gkm_attvar'] = nib.load(datadir+'nomvc_asl_oxford_asl_gkm/native_space/arrival_var.nii.gz').get_fdata().flatten()
            results[protocol+'_nomvc_asl_gkm_attsd'] = np.sqrt(results[protocol+'_nomvc_asl_gkm_attvar'])
            results[protocol+'_nomvc_asl_gkm_attgt'] = nib.load(datadir+'nomvc_gt_att.nii.gz').get_fdata().flatten()
            results[protocol+'_nomvc_asl_gkm_atterr'] = (results[protocol+'_nomvc_asl_gkm_attmean']-results[protocol+'_nomvc_asl_gkm_attgt'])/results[protocol+'_nomvc_asl_gkm_attgt']*100
            results[protocol+'_nomvc_asl_gkm_attabserr'] = np.abs(results[protocol+'_nomvc_asl_gkm_attmean']-results[protocol+'_nomvc_asl_gkm_attgt'])/results[protocol+'_nomvc_asl_gkm_attgt']*100

            results[protocol+'_mvc_asl_gkm_cbfmean'] = nib.load(datadir+'mvc_asl_oxford_asl_gkm/native_space/perfusion.nii.gz').get_fdata().flatten()*cbf_calib
            results[protocol+'_mvc_asl_gkm_cbfvar'] = nib.load(datadir+'mvc_asl_oxford_asl_gkm/native_space/perfusion_var.nii.gz').get_fdata().flatten()*cbf_calib*cbf_calib
            results[protocol+'_mvc_asl_gkm_cbfsd'] = np.sqrt(results[protocol+'_mvc_asl_gkm_cbfvar'])
            results[protocol+'_mvc_asl_gkm_cbfgt'] = nib.load(datadir+'mvc_gt_cbf.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkm_cbferr'] = (results[protocol+'_mvc_asl_gkm_cbfmean']-results[protocol+'_mvc_asl_gkm_cbfgt'])/results[protocol+'_mvc_asl_gkm_cbfgt']*100
            results[protocol+'_mvc_asl_gkm_cbfabserr'] = np.abs(results[protocol+'_mvc_asl_gkm_cbfmean']-results[protocol+'_mvc_asl_gkm_cbfgt'])/results[protocol+'_mvc_asl_gkm_cbfgt']*100


            results[protocol+'_mvc_asl_gkm_attmean'] = nib.load(datadir+'mvc_asl_oxford_asl_gkm/native_space/arrival.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkm_attvar'] = nib.load(datadir+'mvc_asl_oxford_asl_gkm/native_space/arrival_var.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkm_attsd'] = np.sqrt(results[protocol+'_mvc_asl_gkm_attvar'])
            results[protocol+'_mvc_asl_gkm_attgt'] = nib.load(datadir+'mvc_gt_att.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkm_atterr'] = (results[protocol+'_mvc_asl_gkm_attmean']-results[protocol+'_mvc_asl_gkm_attgt'])/results[protocol+'_mvc_asl_gkm_attgt']*100
            results[protocol+'_mvc_asl_gkm_attabserr'] = np.abs(results[protocol+'_mvc_asl_gkm_attmean']-results[protocol+'_mvc_asl_gkm_attgt'])/results[protocol+'_mvc_asl_gkm_attgt']*100

            results[protocol+'_mvc_asl_gkmmvc_cbfmean'] = nib.load(datadir+'mvc_asl_oxford_asl_gkmmvc/native_space/perfusion.nii.gz').get_fdata().flatten()*cbf_calib
            results[protocol+'_mvc_asl_gkmmvc_cbfvar'] = nib.load(datadir+'mvc_asl_oxford_asl_gkmmvc/native_space/perfusion_var.nii.gz').get_fdata().flatten()*cbf_calib*cbf_calib
            results[protocol+'_mvc_asl_gkmmvc_cbfsd'] = np.sqrt(results[protocol+'_mvc_asl_gkmmvc_cbfvar'])
            results[protocol+'_mvc_asl_gkmmvc_cbfgt'] = nib.load(datadir+'mvc_gt_cbf.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkmmvc_cbferr'] = (results[protocol+'_mvc_asl_gkmmvc_cbfmean']-results[protocol+'_mvc_asl_gkmmvc_cbfgt'])/results[protocol+'_mvc_asl_gkmmvc_cbfgt']*100
            results[protocol+'_mvc_asl_gkmmvc_cbfabserr'] = np.abs(results[protocol+'_mvc_asl_gkmmvc_cbfmean']-results[protocol+'_mvc_asl_gkmmvc_cbfgt'])/results[protocol+'_mvc_asl_gkmmvc_cbfgt']*100

            results[protocol+'_mvc_asl_gkmmvc_attmean'] = nib.load(datadir+'mvc_asl_oxford_asl_gkmmvc/native_space/arrival.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkmmvc_attvar'] = nib.load(datadir+'mvc_asl_oxford_asl_gkmmvc/native_space/arrival_var.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkmmvc_attsd'] = np.sqrt(results[protocol+'_mvc_asl_gkmmvc_attvar'])
            results[protocol+'_mvc_asl_gkmmvc_attgt'] = nib.load(datadir+'mvc_gt_att.nii.gz').get_fdata().flatten()
            results[protocol+'_mvc_asl_gkmmvc_atterr'] = (results[protocol+'_mvc_asl_gkmmvc_attmean']-results[protocol+'_mvc_asl_gkmmvc_attgt'])/results[protocol+'_mvc_asl_gkmmvc_attgt']*100
            results[protocol+'_mvc_asl_gkmmvc_attabserr'] = np.abs(results[protocol+'_mvc_asl_gkmmvc_attmean']-results[protocol+'_mvc_asl_gkmmvc_attgt'])/results[protocol+'_mvc_asl_gkmmvc_attgt']*100

            results[protocol+'_mvc_asl_gkmmvc_abv'] = nib.load(datadir+'mvc_asl_oxford_asl_gkmmvc/native_space/aCBV.nii.gz').get_fdata().flatten()*abv_calib
            results[protocol+'_mvc_asl_gkmmvc_abvgt'] = nib.load(datadir+'mvc_gt_abv.nii.gz').get_fdata().flatten()
            index = results[protocol+'_mvc_asl_gkmmvc_abvgt']>0.01
            results[protocol+'_mvc_asl_gkmmvc_abverr'] = np.full(len(results[protocol+'_mvc_asl_gkmmvc_abv']),np.nan)
            results[protocol+'_mvc_asl_gkmmvc_abvabserr'] = np.full(len(results[protocol+'_mvc_asl_gkmmvc_abv']),np.nan)
            results[protocol+'_mvc_asl_gkmmvc_abverr'][index] = (results[protocol+'_mvc_asl_gkmmvc_abv'][index]-results[protocol+'_mvc_asl_gkmmvc_abvgt'][index])/results[protocol+'_mvc_asl_gkmmvc_abvgt'][index]*100
            results[protocol+'_mvc_asl_gkmmvc_abvabserr'][index] = np.abs(results[protocol+'_mvc_asl_gkmmvc_abv'][index]-results[protocol+'_mvc_asl_gkmmvc_abvgt'][index])/results[protocol+'_mvc_asl_gkmmvc_abvgt'][index]*100
            results[protocol+'_mvc_asl_gkmmvc_attagt'] = nib.load(datadir+'mvc_gt_atta.nii.gz').get_fdata().flatten()
            clear_output(wait=True)
            results[protocol+'_mvc_asl_gkmmvc_attagt'] = nib.load(datadir+'mvc_gt_atta.nii.gz').get_fdata().flatten()
            clear_output(wait=True)
            print('Collecting estimation results from protocol',protocol,'(',nProtocol,'of',len(protocolList),') ...')
            print(f'***************  Process: {nProcess/NProcess*100:.2f}%  ***************')
    print('Collecting complete.')

    return results

def get_estimation_avg_oxford_asl(estdir,params,results):

    results_avg = {}
    minPLDs = params['minPLDs']
    totalLDs = params['totalLDs']
    delaylins = params['delaylins']
    for minPLD in enumerate(minPLDs):
        for totalLD in enumerate(totalLDs):
            for delaylin in enumerate(delaylins):
                protocol = f'GEeASL{params["nDelays"]:d}-CV4-{minPLD[1]:.1f}-CV5-{totalLD[1]:.1f}-CV7-{delaylin[1]:.1f}'
                for k1 in ['nomvc_asl_gkm_','mvc_asl_gkm_','mvc_asl_gkmmvc_']:
                    for k2 in ['cbf','att']:
                        for k3 in ['err','abserr']:
                            protocol = f'GEeASL{params["nDelays"]:d}-CV4-{minPLD[1]:.1f}-CV5-{totalLD[1]:.1f}-CV7-{delaylin[1]:.1f}'
                            results_avg[protocol+'_'+k1+k2+k3+'_mean'] = results[protocol+'_'+k1+k2+k3].mean()
                            results_avg[protocol+'_'+k1+k2+k3+'_sd'] = results[protocol+'_'+k1+k2+k3].std()
    return results_avg

def nlls_gkm_rand_func(LDTIs,CBF,ATT,const_M_0a,const_alpha,const_T1_a,const_T1_t,const_lambda):
    signal = np.zeros(len(LDTIs[0]))
    for i in np.arange(len(LDTIs[0])):
        if np.logical_and(LDTIs[1][i]>ATT,LDTIs[1][i]<(ATT+LDTIs[0][i])):
            signal[i] = 2*const_M_0a*const_alpha*CBF/6000/(1/const_T1_t+CBF/6000/const_lambda)*np.exp(-ATT/const_T1_a)*(1-np.exp((ATT-LDTIs[1][i])*(1/const_T1_t+CBF/6000/const_lambda)))
        elif LDTIs[1][i]>(ATT+LDTIs[0][i]):
            signal[i] = 2*const_M_0a*const_alpha*CBF/6000/(1/const_T1_t+CBF/6000/const_lambda)*np.exp(-ATT/const_T1_a)*np.exp((ATT+LDTIs[0][i]-LDTIs[1][i])*(1/const_T1_t+CBF/6000/const_lambda))*(1-np.exp(-LDTIs[0][i]*(1/const_T1_t+CBF/6000/const_lambda)))
    # tRegion1 = np.logical_and(LDTIs[1]>ATT,LDTIs[1]<(ATT+LDTIs[0]))
    # signal[tRegion1] = 2*const_M_0a*const_alpha*CBF/6000/(1/const_T1_t+CBF/6000/const_lambda)*np.exp(-ATT/const_T1_a)*(1-np.exp((ATT-LDTIs[1][tRegion1])*(1/const_T1_t+CBF/6000/const_lambda)))
    # tRegion2 = LDTIs[1]>(ATT+LDTIs[0])
    # signal[tRegion2] = 2*const_M_0a*const_alpha*CBF/6000/(1/const_T1_t+CBF/6000/const_lambda)*np.exp(-ATT/const_T1_a)*np.exp((ATT+LDTIs[0][tRegion2]-LDTIs[1][tRegion2])*(1/const_T1_t+CBF/6000/const_lambda))*(1-np.exp(-LDTIs[0][tRegion2]*(1/const_T1_t+CBF/6000/const_lambda)))
    return signal

def nlls_gkm_rand(params,protocol,CBFgt,ATTgt,noiseSD):

    LDTIs = [protocol['LDs'].tolist(),protocol['TIs'].tolist()]
    CBFest = np.full(CBFgt.shape,np.nan)
    ATTest = np.full(ATTgt.shape,np.nan)

    nProcess = 0
    for i in np.arange(params['nSamp']):
        for j in np.arange(params['nSamp']):
            nProcess += 1
            signal = nlls_gkm_rand_func(LDTIs,CBFgt[i][j],ATTgt[i][j],params['M_0a'],params['labeleff'],params['T1_a'],params['T1_t'],params['lambda'])
            noisy_signal = add_noise(signal,noiseSD)
            nlls_lb = [0.0,0.0,params['M_0a']-0.001,params['labeleff']-0.001,params['T1_a']-0.001,params['T1_t']-0.001,params['lambda']-0.001]
            nlls_ub = [200.0,10.0,params['M_0a'],params['labeleff'],params['T1_a'],params['T1_t'],params['lambda']]
            nlls_init = [60,1.5,params['M_0a'],params['labeleff'],params['T1_a'],params['T1_t'],params['lambda']]
            try:
                popt, _ = curve_fit(nlls_gkm_rand_func,LDTIs,noisy_signal,bounds=(nlls_lb,nlls_ub),p0=nlls_init,method='trf',maxfev=1000)
            except RuntimeError:
                print(f'NLLS fails to estimate at CBF = {CBFgt[i][j]:.1f}, ATT = {ATTgt[i][j]:.1f}, using NaN... ')
                popt = [float('nan'),float('nan')]
            CBFest[i][j] = popt[0]
            ATTest[i][j] = popt[1]
            print(f'************     NLLS     Process: {nProcess/CBFgt.size*100:.2f}%  ************',end='\r')

    CBFerr = (CBFest-CBFgt)/CBFgt*100
    CBFabserr = np.abs(CBFest-CBFgt)/CBFgt*100
    ATTerr = (ATTest-ATTgt)/ATTgt*100
    ATTabserr = np.abs(ATTest-ATTgt)/ATTgt*100
    nlls_CBFerr_mean = np.nanmean(CBFerr.flatten())
    nlls_CBFerr_sd = np.nanstd(CBFerr.flatten())
    nlls_ATTerr_mean = np.nanmean(ATTerr.flatten())
    nlls_ATTerr_sd = np.nanstd(ATTerr.flatten())
    nlls_CBFabserr_mean = np.nanmean(CBFabserr.flatten())
    nlls_CBFabserr_sd = np.nanstd(CBFabserr.flatten())
    nlls_ATTabserr_mean = np.nanmean(ATTabserr.flatten())
    nlls_ATTabserr_sd = np.nanstd(ATTabserr.flatten())

    return nlls_CBFerr_mean,nlls_CBFerr_sd,nlls_ATTerr_mean,nlls_ATTerr_sd,nlls_CBFabserr_mean,nlls_CBFabserr_sd,nlls_ATTabserr_mean,nlls_ATTabserr_sd

def set_dataframe(params):

    df_name = []
    df_nDelays = []
    df_minPLD = []
    df_totalLD = []
    df_delaylin = []
    df_readout = []
    df_scantime = []
    df_isFitBlock = []
    df_nRepeatIn1Block = []
    df_idletime = []
    df_signalSqrtSum = []
    df_wholeSNR = []
    df_tSNR = []
    df_nlls_CBFerr_mean = []
    df_nlls_CBFerr_sd = []
    df_nlls_ATTerr_mean = []
    df_nlls_ATTerr_sd = []
    df_nlls_CBFabserr_mean = []
    df_nlls_CBFabserr_sd = []
    df_nlls_ATTabserr_mean = []
    df_nlls_ATTabserr_sd = []

    noiseSD = set_noise(params,SNR=params['SNR'])
    dim = params['nSamp']
    nomvc_gt_cbf = np.tile(np.linspace(params['cbfRange'][0],params['cbfRange'][0],dim),(dim,1))
    nomvc_gt_att = np.tile(np.reshape(np.linspace(params['attRange'][0],params['attRange'][1],dim),[dim,1]),(1,dim))
    nomvc_gt_atta = np.zeros(nomvc_gt_cbf.shape)
    nomvc_gt_abv = np.zeros(nomvc_gt_cbf.shape)

    nProtocol = 0; NProtocol = len(params['minPLDs'])*len(params['totalLDs'])*len(params['delaylins'])

    for minPLD in params['minPLDs']:
        for totalLD in params['totalLDs']:
            for delaylin in params['delaylins']:
                protocol = set_protocol(params['nDelays'],minPLD,totalLD,delaylin)
                clear_output(wait=True)
                nProtocol += 1
                print('Calculating properties of protocol',protocol['name'],'(',nProtocol,'of',NProtocol,')','...')
                print(f'************  Calculation Process: {nProtocol/NProtocol*100:.2f}%  ************')
                df_name = np.append(df_name,protocol['name'])
                df_nDelays = np.append(df_nDelays,params['nDelays'])
                df_minPLD = np.append(df_minPLD,minPLD)
                df_totalLD = np.append(df_totalLD,totalLD)
                df_delaylin = np.append(df_delaylin,delaylin)
                df_readout = np.append(df_readout,protocol['readout'])
                # df_scantime stores the scantime for 1 repeat
                df_scantime = np.append(df_scantime,protocol['scantime_1rpt'])
                # df_isFitBlock stores if 1 repeat scantime fits in the block design
                isFitBlock = np.logical_and(protocol['scantime_1rpt']>(params['blocktime']-params['blocktaperL']),protocol['scantime_1rpt']<(params['blocktime']+params['blocktaperR']))
                df_isFitBlock = np.append(df_isFitBlock,isFitBlock)
                if isFitBlock:
                    df_nRepeatIn1Block = np.append(df_nRepeatIn1Block,np.floor(params['blocktime']/protocol['scantime_1rpt']))
                    # df_idletime stores the idletime for 1 repeat compared to the block time
                    df_idletime = np.append(df_idletime,(params['blocktime']-protocol['scantime_1rpt']))
                    nHadBlock = protocol['had'].shape[1]
                    signalAcq = np.zeros(nHadBlock)
                    for i in np.arange(nHadBlock):
                        signalAcq[i] = get_signal_rand(params,protocol['LDs'][i],protocol['TIs'][i],nomvc_gt_cbf,nomvc_gt_att,nomvc_gt_atta,nomvc_gt_abv).mean()
                    df_signalSqrtSum = np.append(df_signalSqrtSum,np.sqrt(np.sum(signalAcq)))
                    df_wholeSNR = np.append(df_wholeSNR,np.sqrt(np.sum(signalAcq))/noiseSD)
                    df_tSNR = np.append(df_tSNR,np.sqrt(np.sum(signalAcq))/noiseSD/protocol['scantime_1rpt'])
                    nlls_CBFerr_mean,nlls_CBFerr_sd,nlls_ATTerr_mean,nlls_ATTerr_sd,nlls_CBFabserr_mean,nlls_CBFabserr_sd,nlls_ATTabserr_mean,nlls_ATTabserr_sd = nlls_gkm_rand(params,protocol,nomvc_gt_cbf,nomvc_gt_att,noiseSD)
                    df_nlls_CBFerr_mean = np.append(df_nlls_CBFerr_mean,nlls_CBFerr_mean)
                    df_nlls_CBFerr_sd = np.append(df_nlls_CBFerr_sd,nlls_CBFerr_sd)
                    df_nlls_ATTerr_mean = np.append(df_nlls_ATTerr_mean,nlls_ATTerr_mean)
                    df_nlls_ATTerr_sd = np.append(df_nlls_ATTerr_sd,nlls_ATTerr_sd)
                    df_nlls_CBFabserr_mean = np.append(df_nlls_CBFabserr_mean,nlls_CBFabserr_mean)
                    df_nlls_CBFabserr_sd = np.append(df_nlls_CBFabserr_sd,nlls_CBFabserr_sd)
                    df_nlls_ATTabserr_mean = np.append(df_nlls_ATTabserr_mean,nlls_ATTabserr_mean)
                    df_nlls_ATTabserr_sd = np.append(df_nlls_ATTabserr_sd,nlls_ATTabserr_sd)
                else:
                    df_nRepeatIn1Block = np.append(df_nRepeatIn1Block,np.nan)
                    df_idletime = np.append(df_idletime,np.nan)
                    df_signalSqrtSum = np.append(df_signalSqrtSum,np.nan)
                    df_wholeSNR = np.append(df_wholeSNR,np.nan)
                    df_tSNR = np.append(df_tSNR,np.nan)
                    df_nlls_CBFerr_mean = np.append(df_nlls_CBFerr_mean,np.nan)
                    df_nlls_CBFerr_sd = np.append(df_nlls_CBFerr_sd,np.nan)
                    df_nlls_ATTerr_mean = np.append(df_nlls_ATTerr_mean,np.nan)
                    df_nlls_ATTerr_sd = np.append(df_nlls_ATTerr_sd,np.nan)
                    df_nlls_CBFabserr_mean = np.append(df_nlls_CBFabserr_mean,np.nan)
                    df_nlls_CBFabserr_sd = np.append(df_nlls_CBFabserr_sd,np.nan)
                    df_nlls_ATTabserr_mean = np.append(df_nlls_ATTabserr_mean,np.nan)
                    df_nlls_ATTabserr_sd = np.append(df_nlls_ATTabserr_sd,np.nan)

    protocol_properties = {'name':df_name,'nDelays':df_nDelays,'minPLD (s)':df_minPLD,'totalLD (s)':df_totalLD,'delaylin':df_delaylin,
                           'readout (s)':df_readout,'scantime (s)':df_scantime,'isFitBlock':df_isFitBlock,'nRepeatIn1Block':df_nRepeatIn1Block,'idletime (s)':df_idletime,
                           'signalSqrtSum':df_signalSqrtSum,'wholeSNR':df_wholeSNR,'tSNR':df_tSNR,
                           'nlls_CBFerr_mean (%)':df_nlls_CBFerr_mean,'nlls_CBFerr_sd (%)':df_nlls_CBFerr_sd,'nlls_ATTerr_mean (%)':df_nlls_ATTerr_mean,'nlls_ATTerr_sd (%)':df_nlls_ATTerr_sd,
                           'nlls_CBFabserr_mean (%)':df_nlls_CBFabserr_mean,'nlls_CBFabserr_sd (%)':df_nlls_CBFabserr_sd,'nlls_ATTabserr_mean (%)':df_nlls_ATTabserr_mean,'nlls_ATTabserr_sd (%)':df_nlls_ATTabserr_sd}
    df_protocol_properties = pd.DataFrame(protocol_properties)

    return df_protocol_properties

def visualise_3d_projections(xarray,yarray,zarray,fnxyz,fdiv,xlabels_list,ylabel,plotlabels_list):

    pp.gcf().clear()
    pp.rcParams['figure.figsize'] = [14,3]
    fig = pp.figure(dpi=300)

    ax1 = fig.add_subplot(1,3,1); ax2 = fig.add_subplot(1,3,2); ax3 = fig.add_subplot(1,3,3)
    fxdiv = np.nansum(fdiv,axis=(1,2)); fydiv = np.nansum(fdiv,axis=(0,2)); fzdiv = np.nansum(fdiv,axis=(0,1))

    for i in range(len(fnxyz)):
        fx = np.nansum(fnxyz[i],axis=(1,2))
        ax1.plot(xarray,fx/fxdiv,linewidth=1.0,marker='x',label=plotlabels_list[i])
        fy = np.nansum(fnxyz[i],axis=(0,2))
        ax2.plot(yarray,fy/fydiv,linewidth=1.0,marker='x',label=plotlabels_list[i])
        fz = np.nansum(fnxyz[i],axis=(0,1))
        ax3.plot(zarray,fz/fzdiv,linewidth=1.0,marker='x',label=plotlabels_list[i])

    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3); ax3.grid(alpha=0.3)
    ax1.legend(); ax2.legend(); ax3.legend()
    ylim_min = min([ax1.get_ylim()[0],ax2.get_ylim()[0],ax3.get_ylim()[0]])
    ylim_max = max([ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1]])
    ax1.set_ylim([ylim_min,ylim_max]); ax2.set_ylim([ylim_min,ylim_max]); ax3.set_ylim([ylim_min,ylim_max])
    ax1.set_xlabel(xlabels_list[0]); ax2.set_xlabel(xlabels_list[1]); ax3.set_xlabel(xlabels_list[2])
    ax1.set_ylabel(ylabel); ax2.set_ylabel(ylabel); ax3.set_ylabel(ylabel)

    pp.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=0.30,hspace=0.28)
    
    return fig 













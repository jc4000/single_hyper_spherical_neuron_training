"""
------------------------------
The description of the theory:
-------------------------------
This program is implementation of a hyper-sphere neuron training. 
The output of the neuron is: S(z_i) = 1 / (1 + exp(Alpha*z_i)), where 
z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2 is the square of the distance of input x_i from the surface of the 
hyper-sphere, c_j is a component of the center of the hyper-sphere, R is the radius of the hyper-sphere and 
Alpha > 0 is a scaling factor.

The activation S(z) is mirror reflection of classic sigmoid. S(z) approaches 0 if z approaches infinity and S(z) approaches
1 if z approaches negative infinity.
---------------------------------------------
---------------------------------
Instructions to execute
---------------------------------
Save the file, binary_nand.csv in the same directory where this program is saved

To train with configuration other than the defaults:
Execute the command 'python crclr_single_nrn_entrpy_err.py binary_nand.csv network_config.json'

binary_nand.csv is the file with training data
network_config.json is the file with training configuration details

OR

To train with default configurations:
Execute the command 'python crclr_single_nrn_entrpy_err.py binary_nand.csv'

binary_nand.csv is the file with training data

----------------------------------
Expected Output
----------------------------------
* The convergence status in terms of RMS error and iteration count are displayed on the command shell.
* A file containing the convergence values and the final parameter values are also saved. The file name can be 
  altered in the code at line number 563. The file name starts with 'progress_status'.

"""
#### Libraries
# Standard library
import numpy as np
import math
import csv
import copy
import json
import sys

'''
Read file to collect input data for training

Input Arguments:
        file_Name -- The file name with input data
    
    Output:
        train_inpts  -- input data as 2D array
        train_outpts -- expected outputs as 1D array
        wghts_estmt  -- Estimates of the centers of the data points with label 1 for expected output
        rdius        -- Estimate of radius of the hyper-sphere enclosing the data with label 1 as output
'''
def read_train_input(file_Name):
    train_inpts = [] 
    train_outpts = []
    inpts_estimator = []
    csv_inpt_file = open(file_Name, 'r')
    inpt_readr = csv.reader(csv_inpt_file, delimiter=',', quotechar='|')
    next(inpt_readr, None)
    cnt = 0
    for inpt_rw in inpt_readr:
        #print ('inpt_rw {0}'.format(inpt_rw))
        #print ('target {0}, row :{1}'.format(inpt_rw[-1], cnt))
        cnt = cnt + 1
        if inpt_rw[-1] == '-1':
            train_outpts.append(0.0)
        else:
            train_outpts.append(float(inpt_rw[-1]))
            
        if inpt_rw[-1] == '1':
            inpt_rw.pop(-1)
            inpts_estimator.append(list(map(float, inpt_rw)))
        else:
            inpt_rw.pop(-1)    
        
        train_inpts.append(list(map(float, inpt_rw)))
    
    prdctr_Arr = np.array(inpts_estimator)
    #print ('predictors \n{0}'.format(prdctr_Arr))
    rw_Len = len(prdctr_Arr[0])
    wghts_estmt = []
    rdius = 0.0
    flg = False
    for indx in range (0, rw_Len):
        tmp = []
        for itm in prdctr_Arr:
            tmp.append(itm[indx])
            
        mx = max(tmp)
        mn = min(tmp)
        wghts_estmt.append((float(mx) + float(mn))/2.0)
        dstnc = float(mx - mn)/2.0
        if flg == False:
            rdius = float(mx - mn)/2.0
            flg = True
        else:
            if dstnc > rdius:
                rdius = dstnc
            
    print ('Training Inputs:\n{0}'.format(train_inpts))
    print ('Training Outputs:{0}'.format(train_outpts))
    
    return train_inpts, train_outpts, np.array([wghts_estmt]), np.array([[rdius]])

'''
The class HyperSphereNeuron
'''
class HyperSphereNeuron:  
    '''
    The mirror reflection of the sigmoid function is implemented. 
    If the input to the function is negative then the output is greater than 0.5 and asymptotically approaching 1.
    If the input to the function is positive then the output is less than 0.5 and asymptotically approaching 0.
    
    Input Arguments:
        crcle_Sigz -- The input to the function
        bsts       -- A positive scaling factor to accelerate the asymptotic behavior
    
    Output:
        rt_Sig     -- The output
    '''              
    def crcl_sigmoid(self, crcl_Sigz, bsts):                              #STEP 7                   
        '''The sigmoid function.'''
        
        rt_Sig = 0.0  #Variable to return the output
        exp_Val = 0.0 #Variable to hold the initial computation of the sigmoid
        ovrflw_Flg = False
        #print ('crclr aggregation\n{0}'.format(crcl_Sigz.T[0]))
        #print ('crclr boost\n{0}'.format(bsts))
        #print ('multiplied\n{0}'.format(np.multiply(crcl_Sigz.T, bsts)[0]))
        '''
        Overflow check. Since 0 is the asymptotic minimum, no need to check underflow
        '''
        try:
            exp_Val = np.exp(np.multiply(crcl_Sigz.T, bsts).T)    
        except OverflowError:
            ovrflw_Flg = True
            print ('Overflow error caught for size {0}'.format(crcl_Sigz))
        
        if ovrflw_Flg == False:
            rt_Sig = 1.0/(1.0+exp_Val)
            
        return copy.deepcopy(rt_Sig)
    
    '''
    Compute derivative of the sigmoid 
    Input Arguments:
        acts -- The activations or output of the sigmoid functions
        
    Output:
        ret_Acts -- The output values
    '''
    def crcl_sigmoid_prime(self, acts):
        """Derivative of the sigmoid function."""
        ret_Acts = []
        #print ('crclr acts\n{0}'.format(acts))
        
        np_arr = np.array(acts)
        np_ones = np.ones((len(acts), len(acts[0])))
        #print ('np_arr crcl\n{0}'.format(np_arr))
        #print ('np_ones crcl\n{0}'.format(np_ones))
        ret_Acts.append(np.multiply(np_arr, np.subtract(np_arr, np_ones)))
            
        return copy.deepcopy(ret_Acts)
    
    '''
    Compute activation from the input and the neuron parameters.
    
    Input Arguments:
        bii       -- The bias or radius parameter values
        wee       -- The weight or circle center parameter values
        bst       -- booost or scaling parameter
        crcl_Inpt -- The input to the neuron
    
    
    Outputs:
        crcl_Activation  -- Activations or the neuron output as mirror reflection of sigmoid S(z_i) i=0,1,...
        dstnc            -- The vector result of the input vectors subtracted from the center parameter vector (c_i - x_ij), j=10,...
        crcl_Aggregation -- The square of the radius subtracted from the sum of the square of the components of 
                            the distance vectors z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2, i = 0,1,...; j=0,1,...
    '''
    def crclr_activation(self, bii, wee, bst, crcl_Inpt):
        crcl_Aggregation = []
        #print ('transposed inputs again\n{0}'.format(activations))
        
        crcl_Aggregations = []
        crcl_Activation = None
        #print ('input act\n{0}'.format(crcl_Inpt))
        #print ('weights crclr\n{0}'.format(wee))
        #print ('biases crclr\n{0}'.format(bii))
        #print ('boost\n{0}'.format(bst))
        
        #print ('wee\n{0}'.format(wee))
        #print ('bii\n{0}'.format(bii))
        dstnc = np.subtract(wee, crcl_Inpt)
        #print ('dstnc\n{0}'.format(dstnc))
        #print ('dstnc sqr\n{0}'.format(np.multiply(dstnc, dstnc)))
        #print ('bias sqr\n{0}'.format(np.array((bii*bii))))
        crcl_Aggregation = np.subtract(np.sum(np.multiply(dstnc, dstnc), axis=1), np.array((bii*bii)))
        #print ('dstnc sqr sum - bias sqr\n{0}'.format(crcl_Aggregation))
        crcl_Aggregations.append(copy.deepcopy(crcl_Aggregation))

        crcl_Activation = self.crcl_sigmoid(np.array(crcl_Aggregation), bst)
        
        return copy.deepcopy(crcl_Activation), np.array(copy.deepcopy(dstnc)), copy.deepcopy(crcl_Aggregation)
    
    '''
    Compute the derivative of the crosss entropy error function -[y x Log(S(z))+(1-y) x Log(1 - S(z))].
    The derivative is computed as [y * (1 - S(z)) + (y - 1) * S(z)]. Since y is either 0 or 1, when y = 0 
    then y * (1 - S(z)) = 0 and when y = 1 then (y - 1) * S(z) = 0. Here * means multiplication.
    Input Arguments:
        outpt_Activations -- The activation outputs of the neuron, one output for each row of the 2D input matrix
        trgt_Y            -- The expected target values or y values
    
    Outputs:
        crss_Entrpy_Drvtv -- An array of derivative values, each element of the array is corresponding to the
        row of the 2D input matrix.
    '''
    def crclr_crs_entrpy_drvtv(self, outpt_Activations, trgt_Y):
        np_arr = np.array(outpt_Activations)
        np_ones = np.ones((len(outpt_Activations), len(outpt_Activations[0])))
        
        #print ('crs ntrpy activations\n{0}'.format(np_arr))
        #print ('crs ntrpy drvtv activation ones\n{0}'.format(np_ones))
        
        np_Trgt = np.array([trgt_Y]).transpose()
        np_Trgt_Ones = np.ones((len(trgt_Y), 1))
        
        #print ('crs ntrpy trgts\n{0}'.format(np_Trgt))
        #print ('crs ntrpy drvtv trgt ones\n{0}'.format(np_Trgt_Ones))
        
        true_Outcome_Drvtv = np.multiply(np_Trgt, np.subtract(np_ones, np_arr))
        false_Outcome_Drvtv = np.multiply(np_arr, np.subtract(np_Trgt, np_Trgt_Ones))
        
        #print ('param true side of drvtv\n{0}'.format(true_Outcome_Drvtv))
        #print ('param false side of drvtv\n{0}'.format(false_Outcome_Drvtv))
        
        crss_Entrpy_Drvtv = true_Outcome_Drvtv + false_Outcome_Drvtv
        #print ('cross entropy crclr drvtv\n{0}'.format(crss_Entrpy_Drvtv))
        return copy.deepcopy(crss_Entrpy_Drvtv)
    
    '''
    Compute the gradients corresponding to each input component of the neuron.
    For row i of the input matrix and the corresponding expected output of y_i, 
    the component of the gradient is 2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*(c_j - x_ij). 
    The factor 2.0*(c_j - x_ij) is the derivative of z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2
    with respect to c_j. Alpha > 0 is the scaling factor.
    
    Input ARguments:
        crss_Entrpy_Drvtvs -- The cross entropy factors of the gradients or [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]
        stmls_Drvtvs       -- The factor (C_j - x_ij)s for i = 0,1,2,... and j = 0,1,2,...
        bst_Cmpnnt         -- Scaling parameter or Alpha in the theory
    Outputs:
        crss_Entrpy_Grd    -- The resultant matrix 2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*(c_j - x_ij)
    '''
    def crclr_cross_entrpy_grad(self, crss_Entrpy_Drvtvs, stmls_Drvtvs, bst_Cmpnnt):
        np_ones = np.ones((len(stmls_Drvtvs), len(stmls_Drvtvs[0])))
        np_twos = np.multiply(np_ones, (2.0*bst_Cmpnnt[0][0]))
        
        #print ('crs grad stmls drvtv\n{0}'.format(np_arr))
        #print ('crs grad np twos\n{0}'.format(np_twos))
        np_Stmls_Drvtvs = np.multiply(np.array(stmls_Drvtvs), np_twos)
        
        #print ('stmls multiplied by 2\n{0}'.format(np_Stmls_Drvtvs))
        #print ('entrpy drvtv\n{0}'.format(crss_Entrpy_Drvtvs))
        
        '''
        convert [[x1,x2],\n [x3,x4],\n [x5,x6]] to [[x1,x3,x5],\n [x2,x4,x6]] and 
        also [[x1],\n [x2],\n [x3]] to [[x1,x2,x3]]
        Then multiply each row of the first array by the single row of the second array
        Then transpose back
        '''
        crss_Entrpy_Grd = np.multiply(np_Stmls_Drvtvs.transpose(), crss_Entrpy_Drvtvs.transpose()).transpose()
        
        #print ('crs ntrpy grad\n{0}'.format(crss_Entrpy_Grd))
        return copy.deepcopy(crss_Entrpy_Grd)
    
    '''
    Compute the gradient component for the radius corresponding to each input component of the neuron.
    For row i of the input matrix and the corresponding expected output of y_i, 
    the component of the gradient for the radius is -2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*R. 
    The factor -2.0*R is the derivative of z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2
    with respect to R. Alpha > 0 is the scaling factor.
    
    Input ARguments:
        crss_Entrpy_Drvtvs -- The cross entropy factors of the gradients or [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]
        radius_Drvtv       -- The factor R 
        bst_Cmpnnt         -- Scaling parameter or Alpha in the theory
    Outputs:
        crss_Entrpy_Radius_Grd -- The resultant matrix -2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*R
    '''
    def crclr_cross_entrpy_radius_grad(self, crss_Entrpy_Drvtvs, radius_Drvtv, bst_Cmpnnt):
        np_ones = np.ones((len(crss_Entrpy_Drvtvs), len(crss_Entrpy_Drvtvs[0])))
        np_Radius_Drvtvs = np.multiply(np_ones, ((0.0-2.0)*radius_Drvtv[0][0]*bst_Cmpnnt[0][0]))
        
        crss_Entrpy_Radius_Grd = np.multiply(crss_Entrpy_Drvtvs, np_Radius_Drvtvs)
        
        #print ('crs ntrpy radius drvtv radius\n{0}'.format(np_Radius_Drvtvs))
        #print ('crs ntrpy radius drvtv crs\n{0}'.format(crss_Entrpy_Drvtvs))
        #print ('crs ntrpy radius drvtv\n{0}'.format(crss_Entrpy_Radius_Grd))
        return crss_Entrpy_Radius_Grd
    
    '''
    Compute the gradient component for the scaling factor corresponding to each input component of the neuron.
    For row i of the input matrix and the corresponding expected output of y_i, 
    the component of the gradient for the scaling factor is [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*z_i*1.0. 
    The factor 1.0 is the derivative of z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2
    with respect to Alpha. Alpha > 0 is the scaling factor.
    
    Input ARguments:
        crss_Entrpy_Drvtvs -- The cross entropy factors of the gradients or [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]
        norm_Dstncs        -- The factors, z_i for each i
        
    Outputs:
        crss_Entrpy_Boost_Grd -- The resultant matrix [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*z_i*1.0
    '''
    def crclr_cross_entrpy_boost(self, crs_Ntrpy_Drvs, norm_Dstncs):
        crss_Entrpy_Boost_Grd = np.multiply(crs_Ntrpy_Drvs, np.array(norm_Dstncs).transpose())
        
        #print ('crs ntrpy boost drvtv crs\n{0}'.format(crs_Ntrpy_Drvs))
        #print ('crs ntrpy boost drvtv dstnc nrms\n{0}'.format(norm_Dstncs))
        #print ('crs ntrpy boost drvtv\n{0}'.format(crss_Entrpy_Boost_Grd))
        return crss_Entrpy_Boost_Grd
    
    '''
    Compute root mean square error from the output. Also computes a unique measurement of error that 
    identifies the deviation from the ideal output.
    
    Input ARguments:
        we           -- The parameters for the center of the hyper-sphere of the neuron
        bi           -- The parameters for the radius of the hyper-sphere of the neuron
        bst          -- The parameter for scaling the input to the activation function
        inpt_Mtrx    -- The training input data matrix
        trgt_Outpts  -- The expected training data output
        
    Outputs:
        rms_Err_Cnv    -- A unique measurement of error that identifies the deviation from the ideal output
        t_rms_err      -- Root mean square error
        crcl_Actvation -- S(z_i) i=0,1,...
        crcl_Dstnc     -- (C_j - x_ij) i=0,1,...; j=0,1,...
        crcl_Dstnc_Nrm -- z_i i=0,1,...
    '''
    def compute_tot_rms_err(self, we, bi, bst, inpt_Mtrx, trgt_Outpts):
        crcl_Actvation, crcl_Dstnc, crcl_Dstnc_Nrm = self.crclr_activation(bi, we, bst, inpt_Mtrx)
        
        #self.save_to_csv_file(crcl_Actvation, trgt_Outpts)
        
        np_act_arr = np.array(crcl_Actvation)
        np_trgt_arr = np.array(trgt_Outpts)
        #print ('output activations\n{0}'.format(np_act_arr))
        #print ('target\n{0}'.format(np_trgt_arr.transpose()))
        diff_prims = np.subtract(np_act_arr, np_trgt_arr.transpose())  
        #print ('diffs\n{0}'.format(diff_prims))  
        err_Conv_Lst = []
        for df_Prm in diff_prims[0]:
            if (df_Prm < 0.0):
                if (df_Prm > -0.5):
                    err_Conv_Lst.append([0.0])
                else:
                    err_Conv_Lst.append([df_Prm])
            else:
                if (df_Prm < 0.5):
                    err_Conv_Lst.append([0.0])
                else:
                    err_Conv_Lst.append([df_Prm])
        
        #print ('err conv\n{0}'.format(err_Conv_Lst))
        
        '''
        compute sqrt((err_Conv_Lst DOTPRODUCT err_Conv_Lst) /length of err_Conv_Lst)
        '''
        rms_Err_Cnv = math.sqrt(np.dot(np.array(err_Conv_Lst).transpose(), np.array(err_Conv_Lst)) / len(err_Conv_Lst))
        
        '''
        Compute root mean square error
        '''
        t_rms_err = math.sqrt(np.dot(diff_prims[0].transpose(), diff_prims[0])/max(len(diff_prims[0]), 1.0)) # STEP 12: compute total error
        
        return rms_Err_Cnv, t_rms_err, copy.deepcopy(crcl_Actvation), copy.deepcopy(crcl_Dstnc), copy.deepcopy(crcl_Dstnc_Nrm)
    
    '''
    Compute gradient based changes to the parameters.
    
    Input ARguments:
        crclr_Dstncs     -- C_j - x_ij
        crclr_Dstnc_Nrms -- z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2
        wts              -- The centers of the hype-sphere c_j
        bis              -- Radius R
        bst_Wts          -- Scaling factor Alpha
        crcl_Actvsns     --  S(z)
        trgts            -- Expected output
        
    Outputs:
        crclr_Wt_Grad     -- 2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*(c_j - x_ij)
        crclr_Radius_Grad -- -2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*R
        crclr_Boost_Grad  -- [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]
        avrg_Dnomntr      -- Count of the number of input vectors
    '''
    def crclr_param_grad_changes(self, crclr_Dstncs, crclr_Dstnc_Nrms, wts, bis, bst_Wts, crcl_Actvsns, trgts):
        #print ('param delta w\n{0}'.format(deltas_w))
        #print ('param delta bi\n{0}'.format(deltas_b))
        #print ('param delta bs\n{0}'.format(deltas_bst))
        #print ('param trgts\n{0}'.format(trgts))
        
        '''
        Cross Entropy Derivative [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]
        '''
        crs_Ntrpy_Drv = self.crclr_crs_entrpy_drvtv(crcl_Actvsns, trgts)
        
        '''
        2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*(c_j - x_ij)
        '''
        crclr_Wt_Grad = self.crclr_cross_entrpy_grad(crs_Ntrpy_Drv, crclr_Dstncs, bst_Wts)  
        
        '''
        -2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*R
        '''     
        crclr_Radius_Grad = self.crclr_cross_entrpy_radius_grad(crs_Ntrpy_Drv, bis, bst_Wts)   
        
        '''
        [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*z_i*1.0
        '''    
        crclr_Boost_Grad = self.crclr_cross_entrpy_boost(crs_Ntrpy_Drv, crclr_Dstnc_Nrms)       
        
        '''
        Denominator for computing mean, Count of the number of input vectors
        '''
        avrg_Dnomntr = max(len(crs_Ntrpy_Drv), 1.0)
        #print ('avrg dnmntr:{0}'.format(avrg_Dnomntr))
        
        return copy.deepcopy(crclr_Wt_Grad), copy.deepcopy(crclr_Radius_Grad), copy.deepcopy(crclr_Boost_Grad), copy.deepcopy(avrg_Dnomntr)
    
    '''
    The gradient based changes are normalized by the norm of the components of the gradient based changes
    
    Input ARguments:
        del_W_Grad -- 2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*(c_j - x_ij)
        del_Rdius  -- -2.0*Alpha*[y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*R
        del_Bst    -- [y_i*(1 - S(z_i)) + (y_i - 1)*S(z_i)]*z_i*1.0
        
    Outputs:
        nrmd_Del_W     -- del_W_Grad / sqrt(del_W_Grad**2+del_Rdius**2+del_Bst**2)
        nrmd_Del_Rdius -- del_Rdius / sqrt(del_W_Grad**2+del_Rdius**2+del_Bst**2)
        nrmd_Del_Bst   -- del_Bst / sqrt(del_W_Grad**2+del_Rdius**2+del_Bst**2)
    '''
    def normalized_params(self, del_W_Grad, del_Rdius, del_Bst):
        deltas_w = np.sum(np.multiply(del_W_Grad, del_W_Grad), axis=0)
        #print ('deltas w\n{0}'.format(deltas_w))
        
        deltas_b = np.sum(np.multiply(del_Rdius, del_Rdius), axis=0)
        #print ('deltas radius\n{0}'.format(deltas_b))
        
        deltas_bst = np.sum(np.multiply(del_Bst, del_Bst), axis=0)
        #print ('deltas boost\n{0}'.format(deltas_bst))
        
        grad_norm = math.sqrt(np.sum(deltas_w) + deltas_b[0] + deltas_bst[0])
        #print ('grad_norm\n{0}'.format(grad_norm))
        
        nrmd_Del_W = []
        nrmd_Del_Rdius = []
        nrmd_Del_Bst = []
        
        if grad_norm > 0.0:
            nrmd_Del_W = np.array([np.true_divide(np.sum(del_W_Grad, axis=0), grad_norm)])
            #print ('nrmd del wts\n{0}'.format(nrmd_Del_W))
            nrmd_Del_Rdius = np.array([np.sum(del_Rdius,axis=0)[0] / grad_norm])
            #print ('nrmd del rdius\n{0}'.format(nrmd_Del_Rdius))
            nrmd_Del_Bst = np.array([np.sum(del_Bst, axis=0)[0] / grad_norm])
            #print ('nrmd del bst\n{0}'.format(nrmd_Del_Bst))
        
        return copy.deepcopy(nrmd_Del_W), nrmd_Del_Rdius, nrmd_Del_Bst
    
    '''
    Execute one epoch of the training
    
    Input ARguments:
        write_Fle -- File to write the iteration and rms error
        wts       -- Parameters C_j, the center of the hyper-sphere
        bi        -- Parameter R, the radius of the hyper-sphere
        bst       -- Parameter Alpha, the scaling factor 
        inpt_Mtrx -- Training input matrix
        trgts     -- Expected outputs y
        thrshld   -- Convergence threshold
        itr_cnt   -- Iteration index
        max_itrs  -- Maximum iteration limit
        epsln     -- learning rate
    
    Outputs:
        mod_weights -- modified parameters c_j after the training epoch
        mod_biases  -- modified parameter R after the training epoch
        mod_boosts  -- modified parameter Alpha after the training epoch
        cnvrgd_Flg  -- The root mean square error is less than the thrshld if True, False otherwise
    '''                    
    def train_epoch_sngl_crclr(self, write_Fle, wts, bi, bst, inpt_Mtrx, trgts, thrshld, itr_cnt, max_itrs, epsln):
        activation = np.array(inpt_Mtrx)
        #print ('transposed inputs\n{0}'.format(activation))
        activations = [activation]
        mod_weights = wts 
        mod_biases = bi 
        mod_boosts = bst
        cnvrgd_Flg = False
        '''
        Compute circular hidden layer output
        '''
        rms_Err_Cnv, tot_rms_err, crclr_Actvsns, crclr_Dstncs, crclr_Dstnc_Nrms = self.compute_tot_rms_err(wts, bi, bst, inpt_Mtrx, trgts)
        
        write_Fle.write('iter count {0}\n'.format(itr_cnt))
        write_Fle.write('tot err {0}\n'.format(tot_rms_err))
        #print ('cnv err Lst: {0}'.format(np.array(err_Conv_Lst)))
        print ('cnv rms err: {0}'.format(rms_Err_Cnv))
        print ('tot rms err: {0}'.format(tot_rms_err))
        
        activations.append(np.array(crclr_Actvsns).transpose())
        #print ('crcl sigmoid\n{0}'.format(np.array(crclr_Actvsns).transpose()))
        
        #print ('crcl output\n{0}'.format(crcl_Outpt))
        if (tot_rms_err < thrshld):                             #STEP 13:Stop if error is below threshold
            #print ('activations\n{0}'.format(activations[-1]))
            #print ('aggregations\n{0}'.format(crcl_Aggregations))
            #print ('distances\n{0}'.format(np.array(dstncs)))
            cnvrgd_Flg = True
            print ('cnv check err\n{0}\n Targets\n{1}\n Activations\n{2}'.format(rms_Err_Cnv, trgts, activations[-1]))
        else:
            if itr_cnt < max_itrs: 
                delta_Ws, delta_Rdius, delta_Bst, avrg_Dnmntr = self.crclr_param_grad_changes(crclr_Dstncs, crclr_Dstnc_Nrms, wts, bi, bst, activations[-1], trgts)
                nrm_Del_W, nrm_Del_Rdius, nrm_Del_Bst = self.normalized_params(delta_Ws, delta_Rdius, delta_Bst)
                
                #print ('train epoch wts\n{0}'.format(wts))
                #print ('train epoch nrmd wts\n{0}'.format(nrm_Del_W))
                #print ('train epoch radius\n{0}'.format(bi))
                #print ('train epoch nrmd radius\n{0}'.format(nrm_Del_Rdius))
                #print ('train epoch bst\n{0}'.format(bst))
                #print ('train epoch nrmd bst\n{0}'.format([nrm_Del_Bst]))
                mod_weights = [[w-((epsln*nw)/avrg_Dnmntr) for w, nw in zip(np.array(wts[0]), np.array(nrm_Del_W[0]))]]
                mod_biases = [[b-((epsln*nb)/avrg_Dnmntr) for b, nb in zip(np.array(bi[0]), np.array(nrm_Del_Rdius))]]
                mod_boosts = [[b-((epsln*nb)/avrg_Dnmntr) for b, nb in zip(np.array(bst[0]), np.array(nrm_Del_Bst))]]
                #print ('bias:{0}\n'.format(mod_biases))
                #print ('weights:\n{0}\n'.format(mod_weights))
                #print ('boosts:\n{0}\n'.format(mod_boosts))
                #print ('delta_w:{0}\n'.format(delta_Ws))
                #print ('delta_b:\n{0}\n'.format(delta_Rdius))
                #print ('delta_boosts:\n{0}\n'.format(nrm_Del_Bst))
            else:
                print ('cnv check err\n{0}\n Targets\n{1}\n Activations\n{2}'.format(rms_Err_Cnv, trgts, activations[-1]))
                
        
        return copy.deepcopy(mod_weights), mod_biases, mod_boosts, cnvrgd_Flg
   
if __name__ == "__main__":
    
    '''
    sizes[0] is the number of components of each input data vector
    size[1] is the number of neurons in the hidden layer
    '''
    
    file_Name_prefix = 'progress_status'
    start_Flg = False
    tot_err = 0.0
    
    '''
    Default Configuration values
    '''
    sizes = [2, 1]
    iter_batch = 50000
    run_No = 1                        
    threshold = 0.0001                            
    epsilon = 0.9                              
    max_iters = 1500                            
    iter_cnt = 0                                  
    
    '''
    Commandline inputs for input data file and training configuration
    '''
    inpt_Data_File = ''
    training_Confg_File = ''
    train_Confgs = {}
    
    if len(sys.argv) >= 3:
        inpt_Data_File = sys.argv[1]
        training_Confg_File = sys.argv[2]
        with open(training_Confg_File) as train_Confg_Inpts:
            train_Confgs = json.load(train_Confg_Inpts)
        
        print ('got filenames:{0} {1}'.format(inpt_Data_File, training_Confg_File))
        sizes = [itm for itm in train_Confgs["size"]]
        
        print ('sizes:\n{0}'.format(sizes))
        iter_batch = int(train_Confgs["status_batch_iters"])
        print ('batch status iter:\n{0}'.format(iter_batch))
        run_No = float(train_Confgs["convergence_iter"])
        print ('convergence status file name indexer:\n{0}'.format(run_No))            
        threshold = float(train_Confgs["threshold"])
        print ('Convergence threshold:\n{0}'.format(threshold))                         
        epsilon = float(train_Confgs["epsilon"])
        print ('learning rate:\n{0}'.format(epsilon))                           
        max_iters = int(train_Confgs["max_iter"])   
        print ('Max iterations:\n{0}'.format(iter_batch))                     
        iter_cnt = int(train_Confgs["iter_count"])
        print ('starting of iteration count:\n{0}'.format(iter_cnt))
    else:
        inpt_Data_File = sys.argv[1]
    
    '''
    Prepare file to write convergence details
    '''  
    write_File_Name_Str = 'entropy_{0}_lrn_0_1_conv_0_0001_test'.format(file_Name_prefix)
    write_File_Name='{0}{1}.txt'.format(write_File_Name_Str, run_No)
    write_File = open(write_File_Name, 'w')
    
    '''
    Create object of the class to train a hyper-sphere neuron
    '''
    hdnLyr = HyperSphereNeuron()
    
    '''
    Setup holders for the parameters
    '''
    weights = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
    biases  = [np.zeros((y, 1)) for y in sizes[1:]]
    boosts  = [np.ones((y, 1)) for y in sizes[1:]]
    
    print ('init scaling factor {0}'.format(boosts)) 
    print ('init centers \n{0}'.format(weights))
    print ('init radius\n{0}'.format(biases))
    
    '''
    Read the file with input training data
    '''
    inpt_matrx, trgt_outpt, weights_t, biases_t = read_train_input(inpt_Data_File)
    
    '''
    Initial estimates of the trainning parameters
    '''
    weights[0] = weights_t
    biases[0] = biases_t
    
    print ('centers estimated \n{0}'.format(weights))
    print ('radius estimated\n{0}'.format(biases))
    
    '''
    Start training iterations
    '''    
    while True:
        wght_row = list(weights)
        wght_row.append(list(biases))
        tot_rms_err = 0.0
        #print ('weight with bias \n{0}'.format(wght_row))
            
        diff_prims = None
        
        '''
        An epoch of training
        '''
        weights_, bias_, boost, convrgd = hdnLyr.train_epoch_sngl_crclr(write_File, np.array(weights[0]), np.array(biases[0]), np.array(boosts[0]), inpt_matrx, trgt_outpt, threshold, iter_cnt, max_iters, epsilon)
        
        '''
        Update parameter values with new values computed 
        '''
        weights[0] = np.array(weights_)
        biases[0] = np.array(bias_) 
        boosts[0] = np.array(boost)
        #print ('epoch rslt\n wghts\n{0}, \n bias:{1}, boost:{2}'.format(weights, biases, boosts))
        
        '''
        If training converged or RMS error decreased below the threshold
        '''
        if convrgd == True:
            write_File.write('hidden weights {0}\n'.format(weights[0]))
            write_File.write('hidden radius {0}\n'.format(biases[0]))
            write_File.write('hidden scale {0}\n'.format(boosts[0]))
            write_File.close()
            break
        
        '''
        If did not converge
        '''
        print ('iter:{0}'.format(iter_cnt))
        
        iter_cnt = iter_cnt + 1
        if start_Flg == False:
            start_Flg = True
        elif iter_cnt % iter_batch == 0:
            write_File.write('hidden centers {0}\n'.format(weights[0]))
            write_File.write('hidden radius {0}\n'.format(biases[0]))
            write_File.write('hidden scale {0}\n'.format(boosts[0]))
            write_File.close()
            run_No = run_No + 1
            write_File_Name='{0}{1}.txt'.format(write_File_Name_Str, run_No)
            write_File = open(write_File_Name, 'w')
            
        if iter_cnt >= max_iters:
            write_File.write('hidden centers {0}\n'.format(weights[0]))
            write_File.write('hidden radius {0}\n'.format(biases[0]))
            write_File.write('hidden scale {0}\n'.format(boosts[0]))
        
            write_File.close()
            break
import os
import tensorflow as tf
import keras
import numpy as np
from ER import ER
from DQN import DQN_DENSE
import AGENT_DENSE as AGENT, ENV
import VGG16
import ParticleFilter

# SAM imports, uncomment when using SAM
#from config import *
#from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
#from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss

# LRP imports, uncomment when using LRP
#from captum.attr._core.lrp import LRP
#from captum.attr._core.layer.layer_lrp import LayerLRP

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EPISODES = 1000
MEMORY_SIZE = 25000
REM_STEP = 1
EPSILON = 1.0 # exploration probability at start
EPSILON_MIN = 0.02  # minimum exploration probability
EPSILON_DECAY = 0.00002  # exponential decay rate for exploration prob
BATCH_SIZE = 32
GAMMA = 0.99  # discount rate
NPARTICLES = 250

envs= ["PongNoFrameskip-v4","WizardOfWor-v5","SpaceInvaders-v5","Breakout-v5","Asterix-v5","Seaquest-v5","BeamRider-v5","Qbert-v5"]
env_select=envs[0]

if __name__ == "__main__":
    save_path = "./Models"
    env_name = env_select
    model_path = os.path.join(save_path, env_name + ".h5")
    # VGG16_INPUT_SHAPE = (210, 160, 3) # shape of pong
    VGG16_INPUT_SHAPE = (80, 80, 3) # shape of cropped pong
    INPUT_SHAPE = ( REM_STEP, 2048 ) # shape of dqn

    vgg_model = VGG16.create( input_shape=VGG16_INPUT_SHAPE )

    env = ENV.create( env_name, render=False )
    agent = AGENT.create( REM_STEP,
                          INPUT_SHAPE,
                          env.action_space.n,
                          EPSILON, EPSILON_MIN, EPSILON_DECAY )
    dqn = DQN_DENSE.create( env.action_space.n, INPUT_SHAPE )
    model = dqn["model"]
    target_dqn = DQN_DENSE.create( env.action_space.n, INPUT_SHAPE )
    target_model = target_dqn["model"]
    experiences = ER.create( MEMORY_SIZE )

    DONE = False
    game_steps = 0
    max_average = -21.0
    for e in range( EPISODES ):
        score = 0
        DONE = False
        SAVING = False
        win = 0
        lose = 0

        state = AGENT.reset(
            agent,
            VGG16.predict(
                vgg_model,
                DQN_DENSE.frame2input( ENV.reset( env )[0] )
            ).flatten()
        )
        for _ in range(65):
            ENV.step(env, 0)

        while not DONE:
            print("step: "+str(game_steps))
            game_steps += 1
            action, explore_prob = AGENT.act( agent, model, game_steps )
            frame_rgb, reward, DONE, _ = ENV.step( env, action )
            vgg_output = VGG16.predict(
                vgg_model, DQN_DENSE.frame2input( frame_rgb )
            )
            attFeat = ParticleFilter.selectFiltersByAttention(
                vgg_output, NPARTICLES, model, env
            )


            # SAM model: uncomment when using SAM
            #m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))      
            #m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
            #m.load_weights('SAM/weights/sam-vgg_salicon_weights.pkl')
            #predictionsSaliency = m.predict_generator(generator_test(b_s=b_s, frame_rgb))[0]


            # LRP model: uncomment when using LRP
            #lrp_win_result= ModelFMEvaluation(self,epoch_idx,frame_rgb )
            #max_decomposed_info = np.max(lrp_win_result[0])
            #lrp_win_len = lrp_win_result[0,:,:].shape[0]
            #lrp_weight_vec=np.zeros(lrp_win_len)
            
            #for i, a_win in lrp_win_len:
            #    lrp_weight_vec[i] = predictionsSaliency[a_win]
            
            #lrpTotal=0
            #for a in len(lrp_result):
            #    lrpTotal = lrpTotal+ predictionsSaliency[lrp_result]
            #lrpTotal= lrpTotal/len(lrp_result)
            
            
            #next_state = AGENT.add_frame( agent, np.dot(lrp_weight_vec,attFeat) )
            #VGG16.freeze(vgg_model, np.min(lrp_win_result[:,:,:]))
            



            #next_state = AGENT.add_frame( agent, np.dot(np.expand_dims(attFeat,0,vgg_output.shape[0]),vgg_output )

            next_state = AGENT.add_frame( agent, attFeat )
            
            ER.remember( experiences, state, action, reward, next_state, DONE )
            state = next_state
            score += reward
            if reward != 0.0:
                if reward == 1.0:
                    win += 1
                else:
                    lose += 1
                print( f"{lose}/{win}", end=" " )
            if DONE:
                average = AGENT.set_score_average( agent, score )
                if average >= max_average:
                    max_average = average
                    DQN_DENSE.save( model, model_path )
                    SAVING = True

                print(f"\nepisode: {e}/{EPISODES}, score: {score}, average: {average} e: {explore_prob}, {'SAVING' if SAVING else ''}")

            if game_steps % agent["update_model_steps"] == 0:
                DQN_DENSE.copy_weights( model, target_model )

            DQN_DENSE.train( model, target_model, experiences, BATCH_SIZE, GAMMA )


ACC_TOP=1       
def ModelFMEvaluation(self,epoch_idx,inputs):
        with torch.no_grad():
            self.model.eval()
            flag_temp=self.model.OutputTypeFlag
            #lrp = LRP(self.model)
            layer_lrp = LayerLRP(self.model,self.model.GetIntendedCnvLayer())
            #layer_lrp = LayerLRP(self.model,self.model.classifier[0])
            #for data in self.testloader:
                #inputs, labels = data
                #inputs, labels = inputs.to(self.device),labels.to(self.device)

            self.model.OutputTypeFlag=OutputType.JustOutput
            outputs = self.model(inputs)
            #just for TOP 2 correct prediction output calculate attribution
            idx=torch.zeros(outputs.shape[0],self.ACC_TOP,dtype=torch.int64,requires_grad=False)
            outputs_top_n=outputs.detach().clone()
            for i in range(self.ACC_TOP):
                _,idx[:,i]=torch.max(outputs_top_n,1)
                outputs_top_n[list(range(outputs_top_n.shape[0])),idx[:,i]]=0
            #t=list(range(outputs.shape[0]))
            t=[(y in x)  for x,y in zip(idx,labels)]
            correct_classified_sample=inputs[t,:,:,:]
            correct_classified_labels=labels[t]
            #if(correct_classified_sample.shape[0]==0):
                #continue
            #fm_particle=torch.randint(low=0,high=2,size=(inputs.shape[0],self.model.last_lr_FM_cnt))
            fm_particle=torch.ones((correct_classified_sample.shape[0],self.model.last_lr_FM_cnt))
            #attribution = lrp.attribute(inputs, target=labels,verbose =True)
            #attribution = lrp.attribute(inputs, additional_forward_args=fm_particle,verbose =False)
            #attribution = layer_lrp.attribute(inputs,target=labels,attribute_to_layer_input =True,verbose =False)
            selected_data_idx=0
            self.model.OutputTypeFlag=OutputType.FMSum
            attribution = layer_lrp.attribute(correct_classified_sample,additional_forward_args=fm_particle,attribute_to_layer_input =True,verbose =False)
            high_rel_win=self.FindHighRelevantWin(attribution[0])
            #high_rel_win=self.GetStableWin(correct_classified_sample,correct_classified_labels)
            #self.PlotAttribution_layer(inputs,attribution,labels[0],0,3,epoch_idx)
            #self.PlotAttribution_input(correct_classified_sample,attribution[0],labels[selected_data_idx],selected_data_idx,epoch_idx)
            #self.PlotWinOnImg(correct_classified_sample,high_rel_win,labels[selected_data_idx],selected_data_idx,epoch_idx)
            #print("OK")
        self.model.train()
        self.model.OutputTypeFlag=flag_temp
        return high_rel_win
        
window_sz=6
window_stride=3
window_padding=0
def FindHighRelevantWin(self,input_img_attr):
        #in_img [batch_num,channel,x,y]
        #[(i-k+2P)/s]+1
        #input_img_attr.requires_grad = False
        #input_img_attr = torch.randn(2,3,8,8)
        data_cnt=input_img_attr.shape[0]
        chnl_cnt=input_img_attr.shape[1]
        input_img_attr=torch.reshape(input_img_attr,((input_img_attr.shape[0]*input_img_attr.shape[1]),input_img_attr.shape[2],input_img_attr.shape[3]))
        #each channel is specific data and we do conv specific for it  
        input_img_attr=torch.unsqueeze(input_img_attr,1)#change shape to [batch_num*channel,1,x,y] 
        cnv=nn.Conv2d(1,1,kernel_size=(self.window_sz,self.window_sz), stride=self.window_stride, padding=self.window_padding, dilation=(1,1), bias=False)
        cnv.weight=torch.nn.Parameter(torch.ones((1,1,self.window_sz,self.window_sz)),requires_grad=False)
        res=cnv(input_img_attr)
        res=torch.reshape(res,(data_cnt,chnl_cnt,res.shape[2],res.shape[3]))
        res=torch.clamp(res, min=0)
        res=self.NormalAttrChannel_4Dim(res)
        res=res.sum(1)
        #normalize attribution value before selection
        #res=self.NormalAttrChannel_3Dim(res)
        res_dim=res.shape[1]
        #data_cnt=res.shape[0]=in_img.shape[0]
        t=torch.quantile(res.view(data_cnt,(res_dim*res_dim)),0.85,dim=1)
        #set zero all value less than 75% max
        for i in range(data_cnt):
            res[i,res[i,:,:]<t[i]]=0
        res=self.NormalAttrChannel_3Dim(res)
        #map remaining point in res (max attribution) to corresponding window in input image
        res_window=((res>0).to(dtype=torch.int32)*res).nonzero()
        res_window=torch.cat((res_window,torch.zeros(res_window.shape[0],4)),1)
        res_window[:,3]=res_window[:,1]*self.window_stride#Y cordinate in vision (Top left point)
        res_window[:,4]=res_window[:,2]*self.window_stride#X cordinate in vision
        res_window[:,5]=self.window_sz
        res_window[:,6]=res[res_window[:,0].to(dtype=torch.long),res_window[:,1].to(dtype=torch.long),res_window[:,2].to(dtype=torch.long)]
        return res_window

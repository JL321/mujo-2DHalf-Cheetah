#from gym.envs.mujoco import walker2d
import gym
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.layers import batch_normalization
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
from gym import wrappers

batch_size = 64

max_step = 5000

tf.reset_default_graph()
#tanh output (constricted b/w -1, 1), 6 nodes

#Inputs: 17 dimensional vector

class ReplayBuffer:
    
    def __init__(self, min_fill, max_fill, random_seed = 123):
        
        self.min = min_fill
        self.max = max_fill
        
        self.buffer = deque()
        random.seed(random_seed)
        
    def add(self, s, a, r, s_p, done):
        
        quad_tuple = (s, a, r, s_p, done)
        
        if (len(self.buffer) >= self.max):
        
            self.buffer.popleft()
            self.buffer.append(quad_tuple)

        else:
            
            self.buffer.append(quad_tuple)

    def size(self):
        
        return len(self.buffer)

    def sample(self):

        batch = np.array(random.sample(self.buffer, 64))

        return batch

class Actor:
    
    def __init__(self, bound, obs_space, act_space, scope, target_scope, tau = 0.001):
    
        self.tau = tau
        self.bound = bound
        self.obs_space = obs_space
        self.act_space = act_space
        
        with tf.variable_scope(scope):
            
            
            self.inputs = tf.placeholder(tf.float32, (None, self.obs_space))
            
            Z = fully_connected(self.inputs, 128)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 300)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 400)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, self.act_space, activation_fn = tf.nn.tanh)
            
            self.outputs = tf.multiply(Z, self.bound)
        
        with tf.variable_scope(target_scope):
            
            self.target_inputs = tf.placeholder(tf.float32, (None, self.obs_space))
            
            Z = fully_connected(self.target_inputs, 128)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 300)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 400)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, self.act_space, activation_fn = tf.nn.tanh)
            
            self.target_outputs = tf.multiply(Z, self.bound)
        
        self.params = tf.trainable_variables(scope = scope)
        self.target_params= tf.trainable_variables(scope = target_scope) 
        
        self.action_gradient = tf.placeholder(tf.float32, (None, act_space))
        
        self.unnormalized_actor_gradients = tf.gradients(self.outputs, self.params, -self.action_gradient) #Further clarification required
        self.actor_gradients = list(map(lambda x: tf.div(x, batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(self.actor_gradients, self.params))
        
        self.update_target_network_params = [self.target_params[i].assign(tf.multiply(self.params[i], self.tau)
                                            + tf.multiply(self.target_params[i], (1-self.tau)))
                                            for i in range(len(self.target_params))]
        
    def train(self, state, action_grad):
        
        self.sess.run(self.optimize, feed_dict = {self.inputs: state, self.action_gradient: action_grad})
        
    def predict(self, state):
        
        return self.sess.run(self.outputs, feed_dict = {self.inputs: state})
        
    def target_predict(self, state):
        
        return self.sess.run(self.target_outputs, feed_dict = {self.target_inputs: state})
        
    def update_target(self):
        
        self.sess.run(self.update_target_network_params)
        
    def set_session(self, sess):
        
        self.sess = sess
        
class Critic:
    
    def __init__(self, obs_space, act_space, scope, target_scope, tau = 0.001):
        
        self.obs_space = obs_space
        self.act_space = act_space
        
        self.tau = tau
    
        self.inputs, self.actions, self.outputs = self.create_network(scope)
        self.target_inputs, self.target_actions, self.target_outputs = self.create_network(target_scope)
        
        self.predict_q = tf.placeholder(tf.float32, shape = (None, 1))
        
        self.loss = tf.reduce_mean(tf.square(self.predict_q - self.outputs))
        self.update = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.main_params = tf.trainable_variables(scope)
        
        self.target_params = tf.trainable_variables(target_scope)

        self.update_target_network = [self.target_params[i].assign(tf.multiply(self.main_params[i], self.tau) + tf.multiply(self.target_params[i], 1-self.tau))
                                for i in range(len(self.main_params))]
        
        self.act_gradient = tf.gradients(self.outputs, self.actions)
        
    def create_network(self, scope):
        
       with tf.variable_scope(scope):
           
           inputs = tf.placeholder(tf.float32, shape = (None, self.obs_space))
           actions = tf.placeholder(tf.float32, shape = (None, self.act_space))
           net = fully_connected(inputs, 400)
           net = batch_normalization(net)
           net = tf.nn.relu(net)

           w1 = tf.Variable(tf.random_normal([400, 300]))
           w2 = tf.Variable(tf.random_normal([self.act_space, 300]))  
           b = tf.Variable(tf.zeros([300]))
           
           Z_comb= tf.matmul(net, w1) + tf.matmul(actions, w2) + b

           Z_comb = tf.nn.relu(Z_comb)
           
           outputs = fully_connected(Z_comb, 1, activation_fn = None)
           
           return inputs, actions, outputs
       
    def train(self, state, action, predicted_q):
        
        self.sess.run(self.update, feed_dict = {self.inputs: state, self.actions: action, self.predict_q: predicted_q})
    
    def predict(self, state, action):
        
        return self.sess.run(self.predict_q, feed_dict = {self.inputs: state, self.actions: action})
    
    def update_target(self):
        
        self.sess.run(self.update_target_network)
    
    def predict_target(self, state, action):
        
        return self.sess.run(self.target_outputs, feed_dict = {self.target_inputs: state, self.target_actions : action})
        
    def action_gradient(self, state, action):
        
        return self.sess.run(self.act_gradient, feed_dict = {self.inputs: state, self.actions: action})
        
    def set_session(self, sess):
        
        self.sess = sess
    
def play_one(env, actor, critic, replay_buffer, gamma, explore_noise):
    
    gamma = gamma
    
    state = env.reset()
    done = False
    total_reward = 0
    
    obs_space = actor.obs_space
    act_space = actor.act_space
    
    step = 0
    
    while not done or step >= max_step:
        
        state = np.reshape(state, (1,state.shape[-1]))
        
        action = actor.predict(state) + explore_noise
        
        new_state, reward, done, _ = env.step(action)
        
        new_state = np.reshape(new_state, (1,obs_space))

        total_reward += reward
        replay_buffer.add(state, action, reward, new_state, done)
        state = new_state
        
        state_list = []
        reward_list = []
        statep_list = []
        action_list = []
        done_list = []
        
        multiple_replay = replay_buffer.sample()
        
        for quad_tuple in multiple_replay:
            
            state_list.append(quad_tuple[0])
            reward_list.append(quad_tuple[2])
            statep_list.append(quad_tuple[-2])
            action_list.append(quad_tuple[1])
            done_list.append(quad_tuple[-1])
            
        action_list = np.reshape(np.squeeze(np.array(action_list)),(64,act_space))
        state_list = np.squeeze(np.array(state_list))
        statep_list = np.squeeze(np.array(statep_list))
        reward_list = np.array(reward_list)
        reward_list = np.reshape(reward_list, (64, 1))
        done_list = np.reshape(done_list, (64, 1))
        
        target_qs = []
        
        for i in range(batch_size):
            
            if done_list[i]:
                
                target_qs.append(reward_list[i])
        
            else:
                
                target_qs.append(reward_list[i] + gamma*critic.predict_target(np.reshape(statep_list[i], (1,obs_space)), actor.target_predict(np.reshape(statep_list[i], (1,obs_space)))))
        
        target_qs = np.reshape(np.array(target_qs), (64,1))
  
        critic.train(state_list, action_list, target_qs)
        action_gradient = critic.action_gradient(state_list, actor.predict(state_list))
        actor.train(state_list, action_gradient[0])
        
        step += 1
        
        actor.update_target()
        critic.update_target()
    
    return total_reward

class OrnsteinUhlenbeckActionNoise: #Copied based on the noise used in OpenAI DDPG
    
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

if __name__ == "__main__": 
    
    episode = 500
    min_fill = 100000
    max_fill = 250000

    env = gym.make("HalfCheetah-v2")
    env = wrappers.Monitor(env, "./mujoVid/halfCheetahV", force = True)
    
    bound = env.action_space.high
    
    state_space = env.observation_space.shape[0]
    act_space = env.action_space.shape[0]
    
    gamma = 0.99
    
    replay_buffer = ReplayBuffer(min_fill, max_fill)

    avg_reward = []
        
    actor = Actor(bound, state_space, act_space, "normA", "targetA")
    critic = Critic(state_space, act_space, "normC", "targetC")
    
    saver = tf.train.Saver()
    
    mode = "test"
    
    with tf.Session() as sess:
       
        sess.run(tf.global_variables_initializer())
        
        actor.set_session(sess)
        critic.set_session(sess)

        actor.update_target()
        critic.update_target()
        
        state = env.reset()
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(act_space))
        
        saver.restore(sess, './models/mujoHC.ckpt')
        
        done = False
    
        total_reward = 0
    
        step_c = 0
    
        if mode == "test":
    
            while not done or step_c < 2000:
                
                step_c += 1
                
                env.render()
                   
                state = np.reshape(state, (1,17))
                
                action = actor.predict(state)
                
                state, reward, done, _ = env.step(action)
                
                total_reward += reward
                
            print(total_reward)
        
        
        elif mode == "train":
        
            while replay_buffer.size() < min_fill:
    
                state = np.reshape(state, (1,state_space))
                
                action = np.clip(actor.predict(state) + actor_noise(), -bound, bound)
                new_state, reward, done, _ = env.step(action)
                
                new_state = np.reshape(new_state, (1,state_space))
                
                replay_buffer.add(state, action, reward, new_state, done)
                state = new_state
                
                if done:
                    
                    state = env.reset()
                
            print("Buffer initialized, starting training")
            
            reward_trend = []
            
            for i in range(episode):
    
                total_reward = play_one(env, actor, critic, replay_buffer, gamma, actor_noise())
                
                reward_trend.append(total_reward)
                
                if i % 10 == 0:
                
                    avg_trend = reward_trend[-10:]
                    
                    print(np.mean(avg_trend), " Last 10 rwd ", i)
                
                if i % 100 == 0:
                    
                    avg_trend = reward_trend[-100:]
                    
                    print(np.mean(avg_trend), " Last 100 rwd")
                    
                    print("Saving ..")
                    
                    saver.save(sess, "./models/mujoHC.ckpt")
    
            plt.plot(reward_trend)
            
            plt.show()
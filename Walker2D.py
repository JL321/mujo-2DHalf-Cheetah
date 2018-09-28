#from gym.envs.mujoco import walker2d
import gym
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.layers import batch_normalization
import numpy as np
from collections import deque

batch_size = 32

max_step = 5000

tf.reset_default_graph()
#tanh output (constricted b/w -1, 1), 6 nodes

#Inputs: 17 dimensional vector

class ReplayBuffer:
    
    def __init__(self, min_fill, max_fill):
        
        self.min = min_fill
        self.max = max_fill
        
        self.buffer = deque()
        
    def add(self, s, a, r, s_p):
        
        quad_tuple = (s, a, r, s_p)
        
        self.buffer.append(quad_tuple)
        if (len(self.buffer) > self.max):
        
            self.buffer.pop()

    def size(self):
        
        return len(self.buffer)

    def sample(self):
        
        idx = np.random.randint(0, len(self.buffer))
        return self.buffer[idx]

class Actor:
    
    def __init__(self, scope, target_scope, tau = 0.001):
    
        self.tau = tau
        
        self.action_gradient = tf.placeholder(tf.float32, (None, 6))
        
        with tf.variable_scope(scope):
            
            self.inputs = tf.placeholder(tf.float32, (None, 17))
            
            Z = fully_connected(self.inputs, 128)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 300)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 400)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 6, activation_fn = tf.nn.tanh)
            
            self.outputs = Z
        
        with tf.variable_scope(target_scope):
            
            self.target_inputs = tf.placeholder(tf.float32, (None, 17))
            
            Z = fully_connected(self.target_inputs, 128)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 300)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 400)
            Z = batch_normalization(Z)
            Z = fully_connected(Z, 6, activation_fn = tf.nn.tanh)
            
            self.target_outputs = Z 
        
        self.params = tf.trainable_variables(scope = scope)
        self.target_params= tf.trainable_variables(scope = target_scope) 
        
        self.unnormalized_actor_gradients = tf.gradients(self.outputs, self.params, -self.action_gradient)    
        self.actor_gradients = list(map(lambda x: tf.div(x, batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer().apply_gradients(zip(self.actor_gradients, self.params))
        
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
    
    def __init__(self, scope, target_scope, tau = 0.001):
        
        self.tau = tau
    
        self.inputs, self.actions, self.outputs = self.create_network(scope)
        self.target_inputs, self.target_actions, self.target_outputs = self.create_network(target_scope)
        
        self.predict_q = tf.placeholder(tf.float32, shape = (None))
        
        self.loss = tf.square(self.predict_q - self.outputs)
        self.update = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.main_params = tf.trainable_variables(scope)
        
        self.target_params = tf.trainable_variables(target_scope)

        self.update_target = [self.target_params[i].assign(tf.multiply(self.main_params[i], self.tau) + tf.multiply(self.target_params[i], 1-self.tau))
                                for i in range(len(self.main_params))]
        
        self.act_gradient = tf.gradients(self.outputs, self.actions)
        
    def create_network(self, scope):
        
       with tf.variable_scope(scope):
           
           inputs = tf.placeholder(tf.float32, shape = (None, 17))
           actions = tf.placeholder(tf.float32, shape = (None, 6))

           w1 = tf.Variable(tf.random_normal([128, 300]))
           w2 = tf.Variable(tf.random_normal([6, 300]))  
           b = tf.Variable(tf.zeros([300]))
           
           Z = fully_connected(inputs, 128)
           
           #Z = batch_normalization(Z)

           Z_comb= tf.matmul(Z, w1) + tf.matmul(actions, w2) + b
           Z_comb = tf.nn.relu(Z_comb)
           
           outputs = fully_connected(Z_comb, 1)
           
           return inputs, actions, outputs
       
    def train(self, state, action, predicted_q):
        
        self.sess.run(self.update, feed_dict = {self.inputs: state, self.actions: action, self.predict_q: predicted_q})
    
    def predict(self, state, action):
        
        return self.sess.run(self.predict_q, feed_dict = {self.inputs: state, self.actions: action})
    
    def train_test(self, state, action, predicted_q):
        
        self.sess.run(self.update_target)
    
    def predict_test(self, state, action):
        
        return self.sess.run(self.target_outputs, feed_dict = {self.target_inputs: state, self.target_actions : action})
        
    def action_gradient(self, state, action):
        
        return self.sess.run(self.act_gradient, feed_dict = {self.inputs: state, self.actions: action})
        
    def set_session(self, sess):
        
        self.sess = sess
    
def play_one(env, actor, critic, replay_buffer, gamma):
    
    gamma = gamma
    
    state = env.reset()
    done = False
    total_reward = 0
      
    step = 0
    
    while not done or step >= max_step:
        
        state = np.reshape(state, (1,17))
        
        action = actor.predict(state)
        
        if action.shape != (1,6):
            
            print("TestA")
            print(action)
        
        new_state, reward, done, _ = env.step(action)
        
        new_state = np.reshape(new_state, (1,17))

        total_reward += reward
        replay_buffer.add(state, action, reward, new_state)
        new_state = state
        
        state_list = []
        reward_list = []
        statep_list = []
        action_list = []
        
        for i in range(batch_size):
            
            quad_tuple = replay_buffer.sample()
            
            state_list.append(quad_tuple[0])
            reward_list.append(quad_tuple[2])
            statep_list.append(quad_tuple[-1])
            action_list.append(quad_tuple[1])
            
        action_list = np.squeeze(np.array(action_list))
        state_list = np.squeeze(np.array(state_list))
        statep_list = np.squeeze(np.array(statep_list))
        reward_list = np.array(reward_list)
        
        target_q = reward_list + critic.predict_test(statep_list, action_list)
        critic.train(state_list, action_list, target_q)
        action_gradient = critic.action_gradient(state_list, action_list)
        actor.train(state_list, action_gradient[0])
        
        step += 1
    
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
    
    episode = 1000
    min_fill = 100000
    max_fill = 250000

    env = gym.make("HalfCheetah-v2")
    
    gamma = 0.9
    
    replay_buffer = ReplayBuffer(min_fill, max_fill)

        
    actor = Actor("normA", "targetA")
    critic = Critic("normC", "targetC")
    
    with tf.Session() as sess:
       
        sess.run(tf.global_variables_initializer())
        
        actor.set_session(sess)
        critic.set_session(sess)
        
        saver = tf.train.Saver()
        
        state = env.reset()
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(6))
        
        while replay_buffer.size() < min_fill:

            state = np.reshape(state, (1,17))
            
            action = np.clip(actor.predict(state) + actor_noise(), -1, 1)
            new_state, reward, done, _ = env.step(action)
            
            new_state = np.reshape(new_state, (1,17))
            
            if action.shape != (1,6):
                
                print("Test")
                print(action)
            
            #action = np.squeeze(action)
            
            replay_buffer.add(state, action, reward, new_state)
            state = new_state
            
            if done:
                
                state = env.reset()
            
        print("Buffer initialized, starting training")
            
        reward_trend = []
        
        for i in range(episode):

            total_reward = play_one(env, actor, critic, replay_buffer, gamma)
            
            reward_trend.append(total_reward)
            
            print("OneIter ", total_reward)
            
            if i % 100 == 0:
                
                saver.save(sess, "/models/mujo.ckpt")

        
    
# DISABLE TF WARNING WHEN DEBUGGING
debug = 1
if debug:
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hides INFO + WARNING
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide all messages except errors

from ast_core.distributions.normal import Normal

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

Dx = 2 # Output vector

gauss_dist = Normal(Dx=Dx)

# To avoid TypeError: 'Symbolic Tensor', need to run under
# tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    mu_val, log_sig_val, reg_loss_val, x_val = sess.run([gauss_dist.mu_t,       # mean vector μ
                                                        gauss_dist.log_sig_t,   # log std deviation log(σ)
                                                        gauss_dist.reg_loss_t,  # L2 regularization loss
                                                        gauss_dist.x_t])        # sampled value ~ N(μ, σ²)

print("Mean value             :", mu_val)
print("Log standard deviation :", log_sig_val)
print("Regression loss        :", reg_loss_val)
print("Outputs                :", x_val)

# NEED TO RUN AT LEAST TWICE TO CHECK IF THE SAMPLING IS RANDOM
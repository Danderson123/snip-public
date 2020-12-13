import tensorflow as tf
sess = tf.Session()
# import graph
saver = tf.train.import_meta_graph("./logs/model/itr-9999.meta")

# load weights for graph
saver.restore(sess, "./logs/model/itr-9999")

# get all global variables (including model variables)
vars_global = tf.global_variables()

# get their name and value and put them into dictionary
model_vars = {}
for var in vars_global:
    model_vars[var.name] = var.eval(session=sess)

print(model_vars) #Sanity check- works
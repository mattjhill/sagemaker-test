import tensorflow as tf
import pandas
import pickle
import time
import numpy as np

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_target_mask(size):
  mask = 1 - tf.linalg.set_diag(tf.ones((size, size)), tf.zeros(size - 1), k=1)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2

class InstacartEncoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, embedding_sizes, rate=0.1):
    super(InstacartEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding_layers = [tf.keras.layers.Embedding(size, d_model)
                             for size in embedding_sizes]    

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    # adding embedding and position encoding.
    for i, embedding in enumerate(self.embedding_layers):
      if i == 0:
        x = embedding(x[..., i])
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      else:
        x +=  embedding(x[..., i])

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

def create_masks(inp, tar, causal=True):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp[..., 0])
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp[..., 0])
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  if causal:
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar[..., 0])[1])
  else:
    look_ahead_mask = create_target_mask(tf.shape(tar[..., 0])[1])
  dec_target_padding_mask = create_padding_mask(tar[..., 0])
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1, causal=True):
    super(Transformer, self).__init__()
    self.causal = causal

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training):

    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar, causal=self.causal)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights

class InstacartTransformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, embedding_sizes, rate=0.1):
    super(InstacartTransformer, self).__init__()

    self.encoder = InstacartEncoder(num_layers, d_model, num_heads, dff, 
                           embedding_sizes, rate)

    self.final_layer = tf.keras.layers.Dense(embedding_sizes[0])

  def call(self, inp, training):

    _, look_ahead_mask, _ = create_masks(inp, inp)
    enc_output = self.encoder(inp, training, look_ahead_mask)  # (batch_size, inp_seq_len, d_model)
        
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output


def create_model(num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, is_train=True, rate=0.1, causal=True):
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    internal_model = Transformer(
      num_layers, d_model, num_heads, dff, input_vocab_size, 
      target_vocab_size, pe_input, pe_target, rate=rate, causal=causal
    )
    logits, attn = internal_model(inputs, targets, training=is_train)
    model = tf.keras.Model([inputs, targets], logits)
    return model


def create_decoder_model(num_layers, d_model, num_heads, dff, embedding_sizes, 
                         is_train=True, rate=0.1):
    inputs = tf.keras.layers.Input((None, 4), dtype="int32", name="inputs")
    internal_model = InstacartTransformer(
      num_layers, d_model, num_heads, dff,
      embedding_sizes, rate=rate
    )
    logits = internal_model(inputs, training=is_train)
    model = tf.keras.Model(inputs, logits)
    return model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
  from sklearn.model_selection import train_test_split
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--layers', type=int, default=2)
  parser.add_argument('--d_model', type=int, default=256)
  parser.add_argument('--heads', type=int, default=4)
  parser.add_argument('--dff', type=int, default=256)
  args = parser.parse_args()

  # Set up data pipeline
  BUFFER_SIZE = 20000
  BATCH_SIZE = args.batch_size

  print('loading data')
  products = pandas.read_csv('data/products.csv')
  product_id_max = products['product_id'].max()

  orders_meta = pandas.read_csv('data/orders.csv', index_col='order_id')
  orders_meta = orders_meta[orders_meta['eval_set'] == 'prior']
  orders = pandas.read_csv('data/order_products__prior.csv')
  orders_meta['days_since_first_order'] = orders_meta.groupby('user_id')[['days_since_prior_order']].cumsum()
  orders_meta['days_since_first_order'] = orders_meta['days_since_first_order'].fillna(-1)

  orders_meta[['order_dow', 'order_hour_of_day', 'days_since_first_order']] += 1
  orders = orders.join(orders_meta, on='order_id')
  orders = orders.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'])
  first_orders = orders.groupby('order_id', as_index=False).first()
  first_orders['add_to_cart_order'] = 0
  first_orders['product_id'] = product_id_max + 1
  product_id_max += 1

  last_orders = orders.groupby('order_id', as_index=False).last()
  last_orders['add_to_cart_order'] += 1
  last_orders['product_id'] = product_id_max + 1
  product_id_max += 1

  orders = pandas.concat([first_orders, orders, last_orders])
  orders = orders.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'])
  orders['add_to_cart_order'] += 1
  orders = orders.set_index(['user_id', 'add_to_cart_order'])
  orders = orders[['product_id', 'days_since_first_order', 'order_dow', 'order_hour_of_day']]
  orders_by_user = orders.groupby('user_id')
  def gen_data():
    for o in orders_by_user:
      yield o[1]

  train_dataset = tf.data.Dataset.from_generator(
    gen_data, 
    output_types=(tf.int32), 
    output_shapes=((None, 4))
  )

  bucket_boundaries = [50, 100, 225, 500, 1000, 2000]
  bucket_batch_sizes = [256, 128, 64, 24, 8, 4, 1]

  bucketize = tf.data.experimental.bucket_by_sequence_length(lambda  x: tf.size(x) // 4, bucket_boundaries, bucket_batch_sizes)
  train_dataset = train_dataset.apply(bucketize)
  
  def create_inp_tar(inp):
    return (inp[:, :-1]), inp[:, 1:, 0]
  
  train_dataset = train_dataset.map(create_inp_tar)

  # Define optimization settings
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1
  embedding_sizes = (orders.max().astype(int) + 1).tolist()

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = create_decoder_model(
      num_layers, d_model, num_heads, dff, 
      embedding_sizes,
      rate=dropout_rate, is_train=True
    )
    
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    
    try:
      model.load_weights('data/instacart_transformer.weights')
    except:
      print('could not load weights')

    model.compile(loss=loss_function, optimizer="adam", metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')])

  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'data/instacart_transformer.weights', monitor='loss', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch'
  )

  model.summary()
  model.fit(train_dataset, epochs=10, callbacks=[model_checkpoint])


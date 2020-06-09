import argparse
import json
import os
import pandas
import tensorflow as tf
from transformer_instacart import create_decoder_model, CustomSchedule

def main(args):
    print("running model")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print(os.listdir(args.train))
    print('loading data')
    products = pandas.read_csv(f'{args.train}/products.csv')
    product_id_max = products['product_id'].max()

    orders_meta = pandas.read_csv(f'{args.train}/orders.csv', index_col='order_id')
    orders_meta = orders_meta[orders_meta['eval_set'] == 'prior']
    orders = pandas.read_csv(f'{args.train}/order_products__prior.csv')
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
    bucket_batch_sizes = [4 * x for x in bucket_batch_sizes]

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
        
        # try:
        #     model.load_weights('data/instacart_transformer.weights')
        # except:
        #     print('could not load weights')

        model.compile(loss=loss_function, optimizer="adam", metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')])

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{args.model_output_dir}/instacart_transformer.weights', monitor='loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )

    model.summary()
    model.fit(train_dataset, epochs=10, callbacks=[model_checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts',type=list,default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host',type=str,default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dff', type=int, default=256)

    args = parser.parse_args()

    main(args)

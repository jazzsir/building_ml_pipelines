# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX template complaint prediction model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft

from models import features
from models.keras import constants


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


# 모델이 export되고 deployed되면, 모든 prediction request는 여기 serve_tf_examples_fn을 통해가 된다. 
# 여기서 serialized tf.Example records를 parsing하고 preprocessing steps를 raw request data에 적용한다.
# 이렇게 해서, 모델은 prediction을 preprocessing data로 만든다.
def _get_serve_tf_examples_fn(model, tf_transform_output): 
    """Returns a function that parses a serialized tf.Example and applies TFT.
    """

    model.tft_layer = tf_transform_output.transform_features_layer() # preprocessing graph를 로딩함.

    @tf.function # eager execution은 비용이 많이 든단다. 그래서 성능을 높이고 이식성이 놓게 @tf.function로 그래프로 변환한단다.
    def serve_tf_examples_fn(serialized_tf_examples): # TODO 여기값은 어디서 가져오는지 확인 필요하다. 여기에서 serialized_tf_examples 을 argument로 주는 부분을 못찾았다.
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec() # 이게 실제 ../preprocessing.py 에서 정의한 spec인것 같다.
        feature_spec.pop(features.LABEL_KEY) # 이건 뭐지??
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec # 요청의 raw tf.Example recoreds를 parse 함
        )

        transformed_features = model.tft_layer(parsed_features) # preprocessing trasformation을 raw data에 적용


#        components/module.py에는 아래와 같이 정의되어 있다. 여기서 outputs은 signatures에서 outputs의 이름으로 사용한다. 137p
#        outputs = model(transformed_features)
#        return {"outputs": outputs}

        return model(transformed_features) # preprocessing data로 prediction 수행

    return serve_tf_examples_fn


# 이전 Transform step에서 생성한, compressed되고 preprocessed된 datasets을 load할 수 있게 한다.
# 책(86p)에서는 여기에 tf_transform_output(../preprocessing.py의 return값으로 추측) 으로 data schema를 가져 올 수 있다고 하는데, 
# ../preprocessing.py의 return값 transformed feature operations 이라고 설명 되어 있다. 내가 이해하기론 데이터가 변환되는 방법? 룰?
# 을 정의한거라고 이해 되는데 이걸로 data schema를 볼 수 있는건가? 이게 data schema인가? 음... 근데 data를 변환하려면 schema는 알아야 겠지. schema는 어디에..
def _input_fn(file_pattern, tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern. # 그냥 CSV, TFRecord, Parquet인지를 의미하나?
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy() # transformed된 features에 대한 feature_spec을 가져온다. (이게 ../preprocessing.py에서 만든거인것 같음)
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=features.transformed_name(features.LABEL_KEY), # dataset은 correct batch size로 batched 됨
    )
    
    # 여기 return 값을 generator라고 함.
    return dataset


def get_model(show_summary: bool = True) -> tf.keras.models.Model:
    """
    This function defines a Keras model and returns the model as a Keras object.
    """

    # one-hot categorical features
    input_features = []
    for key, dim in features.ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input( # Input은 keras tensor를 instantiate하는거라네, 여기서 중요한게 name(preprocessing과 일치해야 함)과 shape가 주어짐, 모든 features가 append된다.
                shape=(dim + 1,), name=features.transformed_name(key) # transformed_name은 features.py에 있는거 재사용
            )
        )

    # adding bucketized features
    for key, dim in features.BUCKET_FEATURES.items():
        input_features.append(
            tf.keras.Input(
                shape=(dim + 1,), name=features.transformed_name(key)
            )
        )

    # adding text input features
    input_texts = []
    for key in features.TEXT_FEATURES.keys():
        input_texts.append(
            tf.keras.Input(
                shape=(1,), name=features.transformed_name(key), dtype=tf.string
            )
        )

    # embed text features
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(MODULE_URL) # 모델을 외부에서 가져오네
    reshaped_narrative = tf.reshape(input_texts[0], [-1]) # narrative(뜻:묘사/기술/서술), Keras input은 two-dimensional인데, encoder는 one-dimensional을 원한다네??
    embed_narrative = embed(reshaped_narrative)
    deep_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(
        embed_narrative
    )

    deep = tf.keras.layers.Dense(256, activation="relu")(deep_ff)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dense(16, activation="relu")(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation="relu")(wide_ff)

    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation="sigmoid")(both)

    inputs = input_features + input_texts

    keras_model = tf.keras.models.Model(inputs, output) # model graph와 functional API를 assemble 함
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=constants.LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(),
        ],
    )
    if show_summary:
        keras_model.summary()

    return keras_model


# TFX Trainer will call this function.

# fn_args: trasformation graph, examples datasets, training parameters.
# fn_args.transform_output: ../preprocessing.py의 preprocessing_fn의 return값인것 같다.
# fn_args.train_files and fn_args.eval_files: datasets이겠지.. 이건 train component를 호출할때 줄것 같다.
# 4 steps으로 실행된다. 이 function은 꽤 generic해서 다른 keras 모델에서도 사용할 수 있단다.
def run_fn(fn_args):
    """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
    fn_args: 책에서 보니... trasformation graph, examples datasets, training parameters 등을 포함한다 함.

  """
    # ../preprocessing.py에서 생성한 output을 TFTransformOutput wrapper로 싼다. 이걸로 preprocessing graph도 받아오고 여러가지 제공 함수를 이용한다.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output) 

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, constants.TRAIN_BATCH_SIZE # 1. data generators를 받기 위해 input_fn 호출
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, constants.EVAL_BATCH_SIZE
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = get_model() # 2. compiled Keras model 호출
    # This log path might change in the future.
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps, # 3. 모델 학습, Trainer component(pipeline)에서 받은 수 많은 training/evaluation steps 을 이용
        callbacks=[tensorboard_callback],
    )

    # model signatures를 정의하고 모델을 아래 코드로 저장한다.
    signatures = {
#       여기서 "serving_default"는 signatures에서 signatures Def key로 사용한다. 137p
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
#           여기서 "examples는 signatures에서 inputs 이름으로 사용한다. 137p
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    } # serving function을 포함하는 model signature 정의(8장 135p에 설명)
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

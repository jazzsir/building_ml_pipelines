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
"""TFX complaint prediction preprocessing.

This file defines a template for TFX Transform component.
"""
# 사실 preprocessing 된 값이 어떻게 나올지 감이 오지 않음.
# 찍어보고 싶었는데 못했음. (당연히 dataset을 적용안했으니... 이건 features.py의 내용은 dataset이 아니라 feature를 정의한것임)
# 의문1: raw data에서 어떻게 sparseTensor가 만들어 졌지? 76p에 보면 그냥"Some of our features are of a sparse nature"라고만 되어 있음.

from __future__ import division
from __future__ import print_function

from typing import Union

import tensorflow as tf
import tensorflow_transform as tft

from features import *
#from models import features

# Union type은 추측하기론 둘다 가능하다는것 같다.
def fill_in_missing(x: Union[tf.Tensor, tf.SparseTensor]) -> tf.Tensor:
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a
    dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have
        size at most 1 in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    # isinstance x 가 tf.sparse.SparseTensor 타입인지 check
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0 # tf.string type이면 "" 를 아니면 0 으로 하는듯.
        x = tf.sparse.to_dense( # TFX는 dense를 사용하므로, SparseTensor면 missing data 부분을 default로 설정해서 dense로 변환
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1) # 왜 첫 번째 차원을 삭제한거지? "Args:"에 설명한것 보면 이것의 dence shape가 (1,n)으로 2차원이라고 함, 그래서 1은 의미 없으므로 1은 삭제.


def convert_num_to_one_hot(
    label_tensor: tf.Tensor, num_labels: int = 2
) -> tf.Tensor:
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels) # label_tensor=index, num_lables=dapth
    return tf.reshape(one_hot_tensor, [-1, num_labels]) # 뒤의 값(num_labels)만 고려해서 2차원을 만듦


def convert_zip_code(zipcode: str) -> tf.float32:
    """
    Convert a zipcode string to int64 representation. In the dataset the
    zipcodes are anonymized by repacing the last 3 digits to XXX. We are
    replacing those characters to 000 to simplify the bucketing later on.

    Args:
        str: zipcode
    Returns:
        zipcode: int64
    """
    if zipcode == "":
        zipcode = "00000"
    zipcode = tf.strings.regex_replace(zipcode, r"X{0,5}", "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.float32)
    return zipcode

# return값은 transformed feature operations 이라고 하는데, 이건 keras/model.py 에서(training 단계) TFTransformOutput으로 wrapping 되고 preprocessing graph를 뽑아 쓴다.(hbseo)
def preprocessing_fn(inputs: tf.Tensor) -> tf.Tensor:
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in features.ONE_HOT_FEATURES.keys():
        dim = features.ONE_HOT_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary( # 원래 값은 features.py에 정의한 숫자로 되어 있지 않을것이다. 이걸 이용해서 각 data의 단어를 숫자로 변환해서 최종적으로 one-hot encoding이 가능하게 한다.(내생각)
            fill_in_missing(inputs[key]), top_k=dim + 1 # inputs[key] 에서 inputs은 tensor type이고 tensor는 []안에 스칼라형 tf.Tensor를 인덱스로 사용할 수 있으므로(뭐 배열처럼) key를 이용해서 실제 dataset에 들어 있는 해당 data인것 같다. dim은 key에 대한 값이 들어가므로 features.py에 정의한 값이 들어간다.
        )
        outputs[features.transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
        
    for key, bucket_count in features.BUCKET_FEATURES.items():
        temp_feature = tft.bucketize( # 여긴 10개로 bucketizing하여 들어간다.
            convert_zip_code(fill_in_missing(inputs[key])),
            bucket_count,
            always_return_num_quantiles=False,
        )
        outputs[features.transformed_name(key)] = convert_num_to_one_hot(
            temp_feature, num_labels=bucket_count + 1
        )

    for key in features.TEXT_FEATURES.keys():
        outputs[features.transformed_name(key)] = fill_in_missing(inputs[key])

    outputs[features.transformed_name(features.LABEL_KEY)] = fill_in_missing(
        inputs[features.LABEL_KEY]
    )

    return outputs

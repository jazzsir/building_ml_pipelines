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
"""TFX complaint model model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List

# At least one feature is needed.
# Dataset이 아니라 각 feature를 정의한것이다. 아놔... 바보같이 dataset인줄...
# feature name, feature dimensionality
# One-hot encoding 이므로, 각 features는 숫자 만큼의 길이를 갖는 vector를 갖고 해당하는 data는 그 값만 1로 되고 나머지는 0으로 표시하여 binary category? 로 된다.
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}

# feature name, bucket count
# Buecktizing 은 zip_code를 one-hot encoding하기엔 vector 가 너무 커지므로 10개로 그룹핑함.
BUCKET_FEATURES = {"zip_code": 10}

# feature name, value is unused
TEXT_FEATURES = {"consumer_complaint_narrative": None}

# Keys
LABEL_KEY = "consumer_disputed"

# Transform된 feature를 suffix로 rename함. 이렇게 하는 이유는 .... 정리 했었지?? 뭐 이전 데이터를 참조못하게 한다그랬나?
def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + "_xf"


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + "_vocab"


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]

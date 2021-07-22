"""Kubeflow example using TFX DSL for local deployments (not GCP Cloud AI)."""

import os
import sys

from typing import Text

import absl
from kfp import onprem
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

pipeline_name = "consumer_complaint_pipeline_kubeflow"

persistent_volume_claim = "tfx-pvc"
#persistent_volume = "tfx-pv"
persistent_volume_mount = "/tfx-data"

# temp yaml file for Kubeflow Pipelines
output_filename = f"{pipeline_name}.yaml"
output_dir = os.path.join(
    os.getcwd(), "pipelines", "kubeflow_pipelines", "argo_pipeline_files"
)

# Transform 과 Trainer components를 실행하기 위해 요구되는 Python module code에 대한 file path를 설정한다.
# 또한, raw training data, pipeine artifacts(?), traied model이 저장될 위치등을 세팅힌다.
# 이게 모두 PV에 저장될것 같다.
# pipeline inputs
#data_dir = os.path.join(persistent_volume_mount, "data/consumer_complaints_with_narrative.csv")
data_dir = os.path.join(persistent_volume_mount, "data")
module_file = os.path.join(persistent_volume_mount, "components", "module.py")

# pipeline outputs
output_base = os.path.join(persistent_volume_mount, "output")
serving_model_dir = os.path.join(output_base, pipeline_name)


def init_kubeflow_pipeline(
    components, pipeline_root: Text, direct_num_workers: int
) -> pipeline.Pipeline:

    absl.logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_arg = [f"--direct_num_workers={direct_num_workers}"]
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_arg,
    )
    return p


if __name__ == "__main__":

    absl.logging.set_verbosity(absl.logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        sys.path.append(module_path)

    # default metadata configuration 을 얻는다.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # tfx_image를 custom image를 사용했는데 이 image를 pull못해서 현재 pipline이 안돌아간다..
    tfx_image = os.environ.get(
        "KUBEFLOW_TFX_IMAGE",
        "regi.local:5000/tfx-custom:0.24.0",
        #        "regi.local:5000/tfx-custom:0.1",
    )

    from pipelines.base_pipeline import init_components

    # base_pipelie.py의 init_component를 가져옴.
    components = init_components(
        data_dir,
        module_file,
        serving_model_dir=serving_model_dir,
        training_steps=100,
        eval_steps=100,
    )

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, # default metadata configuration 을 얻는다.
        # Specify custom docker image to use.
        tfx_image=tfx_image,
        pipeline_operator_funcs=(
            # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
            # kubeflow_dag_runner.get_default_pipeline_operator_funcs()
            # provides default configurations specifically for GKE on GCP,
            # such as secrets.
            kubeflow_dag_runner.get_default_pipeline_operator_funcs() # default OpFunc functions 을 얻는다. (OpFunc functions은 summary.md에서 정리했다.)
            + [
                onprem.mount_pvc(pvc_name='tfx-pvc', volume_name='tfx', volume_mount_path='/tfx-data') # 이렇게 추가해서 OpFunc function에 mount한다.
            ]
        ),
    )

    # 이제 runner_config가 준비되었으니, KubeflowDagRunner를 실행시켜서 components를 initialize하자. 이건 바로 kick off되는것이 아니라 runner가 Argo configuration으로 만든다.
    p = init_kubeflow_pipeline(components, output_base, direct_num_workers=0)
    output_filename = f"{pipeline_name}.yaml"
    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config,
        output_dir=output_dir,
        output_filename=output_filename,
    ).run(p)

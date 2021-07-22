import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    ResolverNode,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input


TRAIN_STEPS = 50000
EVAL_STEPS = 10000

# 이상하네 "../base_pipeline.py"도 있는데 이걸 사용하는거여? 아님 저걸 사용하는거여?
# 일단 여기에 주석으로 설명 추가 함.

def init_components(
    data_dir,
    module_file,
    training_steps=TRAIN_STEPS,
    eval_steps=EVAL_STEPS,
    serving_model_dir=None,
    ai_platform_training_args=None,
    ai_platform_serving_args=None,
):

    if serving_model_dir and ai_platform_serving_args:
        raise NotImplementedError(
            "Can't set ai_platform_serving_args and serving_model_dir at "
            "the same time. Choose one deployment option."
        )

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=99
                ),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )
   
    examples = external_input(data_dir)

    example_gen = CsvExampleGen(input=examples, output_config=output)
#    example_gen = CsvExampleGen(input=examples)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    training_kwargs = {
        "module_file": module_file,
        "examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": trainer_pb2.TrainArgs(num_steps=training_steps),
        "eval_args": trainer_pb2.EvalArgs(num_steps=eval_steps),
    }

    if ai_platform_training_args:
        from tfx.extensions.google_cloud_ai_platform.trainer import (
            executor as aip_trainer_executor,
        )

        training_kwargs.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    aip_trainer_executor.GenericExecutor
                ),
                "custom_config": {
                    aip_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args  # noqa
                },
            }
        )
    else:
        training_kwargs.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    GenericExecutor
                )
            }
        )

    trainer = Trainer(**training_kwargs)

    # 이전에 blessing된 model을 metadata store에서 위치 보고 가져온다. 이건 Evaluator component에서 new model이 이전 model과 비교해서 개선되었는지 비교하기 위해 사용함.
    model_resolver = ResolverNode(
        instance_name="latest_blessed_model_resolver",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    )

    # eval_config에 대한 구체적인 내용은 정리 필요. 
    # thresholds부분은 책의 "Valication in the Evaluator Component"에 있어서 정리하였음
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="consumer_disputed")],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["product"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                ],
		# AUC를 0.65이상으로 명시하였고, baseline model(old model) 대비 AUC가 최소 0.01이상으로 되었을 경우 validation 한다. 
		# 다른 어떠한 metrics도 AUC 부분에 추가할 수 있다. 
		# 다만, 추가하고자 하는 metrics은 필수로 <>MetricsSpec<>안에 <>metrics<> 목록안에 추가되어야 한다.
                thresholds={
                    "AUC": tfma.config.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.65}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.01},
                        ),
                    )
                },
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"], # validation dataset 일듯
        model=trainer.outputs["model"], # new model
        baseline_model=model_resolver.outputs["model"], # 이전 model
        eval_config=eval_config,
    )

    pusher_kwargs = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
    }

    if ai_platform_serving_args:
        from tfx.extensions.google_cloud_ai_platform.pusher import (
            executor as aip_pusher_executor,
        )

        pusher_kwargs.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    aip_pusher_executor.Executor
                ),
                "custom_config": {
                    aip_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args  # noqa
                },
            }
        )
    elif serving_model_dir:
        pusher_kwargs.update(
            {
                "push_destination": pusher_pb2.PushDestination(
                    filesystem=pusher_pb2.PushDestination.Filesystem(
                        base_directory=serving_model_dir
                    )
                )
            }
        )
    else:
        raise NotImplementedError(
            "Provide ai_platform_serving_args or serving_model_dir."
        )

    pusher = Pusher(**pusher_kwargs)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    return components

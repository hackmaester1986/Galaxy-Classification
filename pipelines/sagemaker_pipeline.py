import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel as RegisterModelStepCollection
from sagemaker.workflow.steps import CacheConfig

def get_pipeline(
    region: str,
    role: str,
    default_bucket: str,
    image_uri: str,
    pipeline_name: str = "galaxy-classifier-pipeline",
    base_job_prefix: str = "galaxy-classifier",
):
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=default_bucket,
    )

    # ---------- Parameters ----------
    input_csv_s3_uri = ParameterString(
        name="InputCsvS3Uri",
        default_value=f"s3://{default_bucket}/galaxy-classifier/data/processed/all_data_s3.csv",
    )

    image_uri_param = ParameterString(
        name="ImageUri",
        default_value=image_uri,
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )

    instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.g4dn.2xlarge",
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge",
    )

    epochs_stage1 = ParameterInteger(
        name="EpochsStage1",
        default_value=2,
    )

    epochs_stage2 = ParameterInteger(
        name="EpochsStage2",
        default_value=5,
    )

    batch_size = ParameterInteger(
        name="BatchSize",
        default_value=32,
    )

    # ---------- Training Step ----------
    estimator = Estimator(
        image_uri=image_uri_param,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{default_bucket}/galaxy-classifier/models/",
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}-train",
        hyperparameters={
            "input-csv": "/opt/ml/input/data/train",
            "model-dir": "/opt/ml/model",
            "epochs-stage1": epochs_stage1,
            "epochs-stage2": epochs_stage2,
            "batch-size": batch_size,
            "num-workers": 0,
        },
    )

    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="30d"
    )

    train_step = TrainingStep(
        name="TrainGalaxyClassifier",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=input_csv_s3_uri,
                content_type="text/csv",
            )
        },
        cache_config=cache_config
    )

    # ---------- Evaluation Step ----------
    eval_processor = ScriptProcessor(
        image_uri=image_uri_param,
        command=["python"],
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}-eval",
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="metrics.json",
    )

    eval_step = ProcessingStep(
        name="EvaluateGalaxyClassifier",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=input_csv_s3_uri,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{default_bucket}/galaxy-classifier/evaluations/",
            )
        ],
        code="scripts/evaluate_models.py",
        job_arguments=[
            "--input-csv", "/opt/ml/processing/input/data",
            "--model-dir", "/opt/ml/processing/model",
            "--eval-dir", "/opt/ml/processing/evaluation",
            "--batch-size", batch_size.to_string(),
            "--num-workers", "2",
        ],
        property_files=[evaluation_report],
    )

    # ---------- Register Model ----------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_step.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            content_type="application/json",
        )
    )

    register_step = RegisterModelStepCollection(
        name="RegisterGalaxyClassifierModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-image"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="galaxy-classifier-model-group",
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # ---------- Metric Gate ----------
    cond_step = ConditionStep(
        name="AccuracyGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=evaluation_report,
                    json_path="stage2_resnet_accuracy",
                ),
                right=0.70,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_csv_s3_uri,
            image_uri_param,
            model_approval_status,
            instance_type,
            processing_instance_type,
            epochs_stage1,
            epochs_stage2,
            batch_size,
        ],
        steps=[
            train_step,
            eval_step,
            cond_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


if __name__ == "__main__":
    region = boto3.Session().region_name or "us-east-1"
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker.Session().default_bucket()

    image_uri = "<YOUR_ECR_IMAGE_URI>"
    pipeline = get_pipeline(
        region=region,
        role=role,
        default_bucket=default_bucket,
        image_uri=image_uri,
    )

    pipeline.upsert(role_arn=role)
    print(f"Upserted pipeline: {pipeline.name}")
# Historical deployment evidence — separate implementations

The following are screenshots the project author supplied. They are not generated demonstrations and do not establish current cloud availability.

| Evidence | File | Interpretation |
| --- | --- | --- |
| MLflow dashboard | `screenshots/mlflow-experiment-runs.webp` | Multiple baseline and optimized experiment runs |
| Local Streamlit | `screenshots/streamlit-local-prediction.webp` | A local form produces Transported = True |
| AWS SageMaker Notebook | `screenshots/sagemaker-training-optuna.webp` | Separate Logistic Regression optimization execution; CV ~0.7927 |
| AWS SageMaker Endpoint | `screenshots/sagemaker-endpoint-inservice.webp` | InService at screenshot time, May 19, 2026 |
| AWS EC2 | `screenshots/aws-ec2-instance.webp` | A running cloud instance / health checks at test time |
| Cloud Streamlit | `screenshots/cloud-streamlit-form.webp` | Form hosted on EC2 |
| SageMaker response | `screenshots/cloud-sagemaker-prediction.webp` | Payload and returned Transported=True, P(class 1) ~0.7407 |

AWS screenshots have been cropped or masked to avoid publishing account identities, endpoint ARNs and network addresses. The root Python source files here are an earlier/local deployment variant; complete scripts for provisioning the shown AWS resources were not supplied for this portfolio update. Do not claim the repo is a full infrastructure-as-code solution or a currently reachable cloud demo.

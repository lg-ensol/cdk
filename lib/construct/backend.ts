import * as cdk from "aws-cdk-lib";
import * as path from "path";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as apigateway from "aws-cdk-lib/aws-apigateway";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";

export interface BackendProps {
  userPool: cognito.UserPool;
  table: dynamodb.Table;
}

export class BackendConstruct extends Construct {
  public readonly apiUrl: string;

  constructor(scope: Construct, id: string, props: BackendProps) {
    super(scope, id);

    // Lambda Insights 권한 추가
    const insightsPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
      ],
      resources: ["*"],
    });

    // Lambda 레이어 생성
    const lambdaLayer = new lambda.LayerVersion(this, "AWSSDK-Layer", {
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "layer", "layer.zip")
      ),
      compatibleRuntimes: [lambda.Runtime.NODEJS_20_X],
      description: "A layer for using AWS SDK in  Lambda function",
    });

    const pythonLayer = new lambda.LayerVersion(this, "boto3-Layer", {
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "layer", "boto3.zip")
      ),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_11],
      description: "A layer for using AWS SDK in  Lambda function",
    });

    // Lambda 함수 생성
    const salesLambda = new lambda.Function(this, "SalesLambda", {
      functionName: "SalesLambda",
      runtime: lambda.Runtime.NODEJS_20_X,
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "lambda")
      ),
      handler: "serverless.handler",
      layers: [lambdaLayer],
      insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_119_0, // Lambda Insights 활성화
      memorySize: 512,
    });

    salesLambda.addToRolePolicy(insightsPolicy);

    const publicLambda = new lambda.Function(this, "PublicHandler", {
      runtime: lambda.Runtime.NODEJS_20_X,
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "lambda")
      ),
      handler: "public.handler",
      layers: [lambdaLayer],
      insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_119_0, // Lambda Insights 활성화
      memorySize: 512,
    });

    publicLambda.addToRolePolicy(insightsPolicy);

    const invokeLambda = new lambda.Function(this, "InvokeHandler", {
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "lambda")
      ),
      handler: "invoke.lambda_handler",
      layers: [pythonLayer],
      insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_119_0, // Lambda Insights 활성화
      memorySize: 512,
      });

    invokeLambda.addToRolePolicy(insightsPolicy);

    const dynamoDbPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
      ],
      resources: [props.table.tableArn],
    });

    const kanbanLambda = new lambda.Function(this, "KanbanHandler", {
      runtime: lambda.Runtime.NODEJS_20_X,
      code: lambda.Code.fromAsset(
        path.join(__dirname, "..", "..", "src", "lambda")
      ),
      handler: "kanban.handler",
      layers: [lambdaLayer],
      insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_119_0, // Lambda Insights 활성화
      memorySize: 512,
    });

    kanbanLambda.addToRolePolicy(insightsPolicy);
    kanbanLambda.addToRolePolicy(dynamoDbPolicy);

    // API Gateway 생성 및 CORS 설정
    const api = new apigateway.RestApi(this, "ServerlessAPI", {
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: [
          "Content-Type",
          "X-Amz-Date",
          "Authorization",
          "X-Api-Key",
          "X-Amz-Security-Token",
        ],
        allowCredentials: true,
      },
    });

    // Lambda 통합 생성
    const salesIntegration = new apigateway.LambdaIntegration(salesLambda);
    const publicIntegration = new apigateway.LambdaIntegration(publicLambda);
    const kanbanIntegration = new apigateway.LambdaIntegration(kanbanLambda);
    const invokeIntegration = new apigateway.LambdaIntegration(invokeLambda);

    // Cognito Authorizer 생성
    const authorizer = new apigateway.CognitoUserPoolsAuthorizer(
      this,
      "CognitoAuthorizer",
      {
        cognitoUserPools: [props.userPool],
      }
    );

    // 리소스 및 메서드 추가
    const salesResource = api.root.addResource("sales");
    salesResource.addMethod("GET", salesIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    const publicResource = api.root.addResource("public");
    publicResource.addMethod("GET", publicIntegration);

    const kanbanResource = api.root.addResource("kanban");
    kanbanResource.addMethod("GET", kanbanIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    const invokeResource = api.root.addResource("invoke");
    invokeResource.addMethod("POST", invokeIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    this.apiUrl = api.url;

    // API Gateway URL을 CloudFormation 출력으로 내보내기
    new cdk.CfnOutput(this, "ApiGatewayUrl", {
      value: api.url,
      description: "API Gateway URL",
      exportName: "ApiGatewayUrl",
    });
  }
}

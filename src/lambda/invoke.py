# -*- coding: utf-8 -*-

import json
import logging
import boto3
import os
from collections import OrderedDict
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log(message):
    logger.info(message)

AGENT_ID = os.environ["AGENT_ID"]
AGENT_ALIAS_ID = os.environ["AGENT_ALIAS_ID"]
REGION_NAME = os.environ["REGION_NAME"]

log(f"Agent id: {AGENT_ID}")
log(f"Agent Alias id: {AGENT_ALIAS_ID}")

agent_client = boto3.client("bedrock-agent", region_name=REGION_NAME)
agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=REGION_NAME)
s3_resource = boto3.resource("s3", region_name=REGION_NAME)

def get_highest_agent_version_alias_id(response):
    highest_version = None
    highest_version_alias_id = None
    for alias_summary in response.get("agentAliasSummaries", []):
        if alias_summary["routingConfiguration"]:
            agent_version = alias_summary["routingConfiguration"][0]["agentVersion"]
            if agent_version.isdigit() and (highest_version is None or int(agent_version) > highest_version):
                highest_version = int(agent_version)
                highest_version_alias_id = alias_summary["agentAliasId"]
    return highest_version_alias_id

def invoke_agent(user_input, session_id):
    log(f"User input response... {user_input}")
    streaming_response = agent_runtime_client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        enableTrace=True,
        inputText=user_input,
    )
    return streaming_response

def get_agent_response(response):
    log(f"Getting agent response... {response}")
    if "completion" not in response:
        return f"No completion found in response: {response}"
    trace_list = []
    for event in response["completion"]:
        log(f"Event keys: {event.keys()}")
        if "trace" in event:
            log(event["trace"])
            trace_list.append(event["trace"])
        if "chunk" in event:
            chunk_bytes = event["chunk"]["bytes"]
            chunk_text = chunk_bytes.decode("utf-8")
            print("Response from the agent:", chunk_text)
    sql_query_from_llm = None
    for t in trace_list:
        if "orchestrationTrace" in t["trace"].keys():
            if "observation" in t["trace"]["orchestrationTrace"].keys():
                obs = t["trace"]["orchestrationTrace"]["observation"]
                if obs["type"] == "ACTION_GROUP":
                    sql_query_from_llm = extract_sql_query(obs["actionGroupInvocationOutput"]["text"])
    if sql_query_from_llm:
        source_file_list = sql_query_from_llm
    else:
        try:
            source_file_list = extract_source_list_from_kb(trace_list)
        except Exception as e:
            log(f"Error extracting source list from KB: {e}")
            source_file_list = ""
    return chunk_text, source_file_list

def extract_source_list_from_kb(trace_list):
    for trace in trace_list:
        if 'orchestrationTrace' in trace['trace'].keys() and 'observation' in trace['trace']['orchestrationTrace'].keys():
            if 'knowledgeBaseLookupOutput' in trace['trace']['orchestrationTrace']['observation']:
                ref_list = trace['trace']['orchestrationTrace']['observation']['knowledgeBaseLookupOutput']['retrievedReferences']
                log(f"ref_list: {ref_list}")
                ref_s3_list = []
                for rl in ref_list:
                    ref_s3_list.append(rl['location']['s3Location']['uri'])
                return ref_s3_list

def source_link(input_source_list):
    source_dict_list = []
    for i, input_source in enumerate(input_source_list):
        string = input_source.split("//")[1]
        bucket = string.partition("/")[0]
        obj = string.partition("/")[2]
        file = s3_resource.Object(bucket, obj)
        body = file.get()["Body"].read()
        try:
            decoded_body = body.decode('utf-8')
            res = json.loads(decoded_body)
            source_link_url = res["Url"]
            source_title = res["Topic"]
            source_dict = (source_title, source_link_url)
            source_dict_list.append(source_dict)
        except UnicodeDecodeError: 
            pass
    unique_sources = list(OrderedDict.fromkeys(source_dict_list))
    refs_str = ""
    for i, (title, link) in enumerate(unique_sources, start=1):
        refs_str += f"{i}. [{title}]({link})\n\n"
    return refs_str

def extract_sql_query(input_string):
    pattern = r"(SELECT.*?)(?=\n\s*(?:Returned information|$))"
    match = re.search(pattern, input_string, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None

def lambda_handler(event, context):
    log("Event:")
    log(json.dumps(event))

    # Directly access the body as it is already a dictionary
    body = json.loads(event["body"])

    streaming_response = invoke_agent(body["query"], body["session_id"])
    response, source_file_list = get_agent_response(streaming_response)

    if isinstance(source_file_list, list):
        reference_str = source_link(source_file_list)
    else:
        reference_str = source_file_list

    print(f"reference_str: {reference_str}")

    output = {"answer": response, "source": reference_str}

    return {'statusCode': 200,'body': json.dumps(output)}

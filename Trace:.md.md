Trace:
{"prompts":[{"events":[{"field":"input","name":"data.input","roles":["system","user","assistant"]},{"field":"response","name":"data.output","roles":["system","user","assistant"]}],"span_types":["inference","inference.framework","agentic.tool.invocation","agentic.invocation","embedding","embedding.modelapi","retrieval","retrieval.embedding"]}]}

Session:
{"prompts":[{"events":[{"field":"input","name":"data.input","roles":["system","user","assistant"]},{"field":"response","name":"data.output","roles":["system","user","assistant"]}],"span_types":["inference","inference.framework","agentic.tool.invocation","agentic.invocation","embedding","embedding.modelapi","retrieval","retrieval.embedding"]}]}


to keep the relationship we need to inlcude some of the attributes such as inference.decision.span.id: 70c1a010e9a4a554


curl -i -H "x-api-key: $OKAHU_API_KEY" \ "https://eval.okahu.co/api/v1/eval/templates?fact_name=traces"
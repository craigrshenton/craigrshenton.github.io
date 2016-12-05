{% extends 'markdown.tpl' %}

{%- block header -%}
---
layout: post
title: "{{resources['metadata']['name']}}"
tags:
    - python
    - notebook
---
{%- endblock header -%}

{% block in_prompt %}
**In [{{ cell.execution_count }}]:**
{% endblock in_prompt %}

{% block input %}
{{ '```python' }}
{{ cell.source }}
{{ '```' }}
{% endblock input %}

---
layout: page
title: "Articles"
permalink: /blog/
---

{% for post in site.posts %}
<div style="padding: 1rem 0; border-bottom: 1px solid #e8e8e8;">
  <div style="font-size: 0.8rem; color: #999; margin-bottom: 0.25rem;">{{ post.date | date: "%d %b %Y" }}</div>
  <a href="{{ post.url }}" style="font-weight: 600; font-size: 1rem; color: #1a1d2e;">{{ post.title }}</a>
  {% if post.excerpt %}
  <p style="font-size: 0.9rem; color: #555; margin: 0.4rem 0 0; line-height: 1.5;">{{ post.excerpt | strip_html | truncate: 180 }}</p>
  {% endif %}
</div>
{% endfor %}

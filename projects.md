---
title: "Project Experience"
layout: archive
permalink: /projects/
author_profile: true
---

## Featured Projects

{% for post in site.posts %}
  <article class="archive__item" style="margin-bottom: 2em; border-bottom: 1px solid #eee; padding-bottom: 1em;">
    <h3 class="archive__item-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
    <p class="page__meta"><i class="far fa-calendar-alt"></i> {{ post.date | date: "%B %d, %Y" }}</p>
    {% if post.excerpt %}<p class="archive__item-excerpt">{{ post.excerpt | strip_html | truncate: 160 }}</p>{% endif %}
  </article>
{% endfor %}

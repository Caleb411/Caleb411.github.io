{% extends '_layout.swig' %}
{% import '_macro/sidebar.swig' as sidebar_template with context %}

{% block title %}
  {%- set page_title_suffix = ' | ' + title %}

  {%- if page.type === 'categories' and not page.title %}
    {{- __('title.category') + page_title_suffix }}
  {%- elif page.type === 'tags' and not page.title %}
    {{- __('title.tag') + page_title_suffix }}
  {%- elif page.type === 'schedule' and not page.title %}
    {{- __('title.schedule') + page_title_suffix }}
  {%- else %}
    {{- page.title + page_title_suffix }}
  {%- endif %}
{% endblock %}

{% block class %}page posts-expand{% endblock %}

{% block content %}

    {##################}
    {### PAGE BLOCK ###}
    {##################}
    <div class="post-block" lang="{{ page.lang or config.language }}">
      {% include '_partials/page/page-header.swig' %}
      {#################}
      {### PAGE BODY ###}
      {#################}
      <div class="post-body{%- if page.direction and page.direction.toLowerCase() === 'rtl' %} rtl{%- endif %}">
        {%- if page.type === 'tags' %}
		<div class="tag-cloud">
		  <!-- <div class="tag-cloud-title">
			  {{ _p('counter.tag_cloud', site.tags.length) }}
		  </div> -->
		  <div class="tag-cloud-tags" id="tags">
			{{ tagcloud({min_font: 16, max_font: 16, amount: 300, color: true, start_color: '#fff', end_color: '#fff'}) }}
		  </div>
		</div>
		<br>

		<script type="text/javascript">
		   var alltags=document.getElementById('tags');
		   var tags=alltags.getElementsByTagName('a');

		   for (var i = tags.length - 1; i >= 0; i--) {
			 var r=Math.floor(Math.random()*75+130);
			 var g=Math.floor(Math.random()*75+100);
			 var b=Math.floor(Math.random()*75+80);
			 tags[i].style.background = "rgb("+r+","+g+","+b+")";
		   }
		</script>

		<style type="text/css">
			div#posts.posts-expand .tag-cloud a{
		   background-color: #f5f7f1;
		   border-radius: 6px;
		   padding-left: 10px;
		   padding-right: 10px;
		   margin-top: 18px;

		 }

		 .tag-cloud a{
		   background-color: #f5f7f1;
		   border-radius: 4px;
		   padding-right: 5px;
		   padding-left: 5px;
		   margin-right: 5px;
		   margin-left: 0px;
		   margin-top: 8px;
		   margin-bottom: 0px;

		 }

		 .tag-cloud a:before{
			  content: "";
		 }

		 .tag-cloud-tags{
		   text-align: middle;
		   counter-reset: tags;
		 }
		</style>
        {% elif page.type === 'categories' %}
          <div class="category-all-page">
            <div class="category-all-title">
              {{ _p('counter.categories', site.categories.length) }}
            </div>
            <div class="category-all">
              {{ list_categories() }}
            </div>
          </div>
        {% elif page.type === 'schedule' %}
          <div class="event-list">
          </div>
          {% include '_scripts/pages/schedule.swig' %}
        {% else %}
          {{ page.content }}
        {%- endif %}
      </div>
      {#####################}
      {### END PAGE BODY ###}
      {#####################}
    </div>
    {% include '_partials/page/breadcrumb.swig' %}
    {######################}
    {### END PAGE BLOCK ###}
    {######################}

{% endblock %}

{% block sidebar %}
  {{ sidebar_template.render(true) }}
{% endblock %}

#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Mohcine Madkour'
SITENAME = u'Mohcine Madkour'
TAGLINE = u'Big Data Architectures and more'
SITEURL = 'https://mohcinemadkour.github.io'

PATH = 'content'

TIMEZONE = 'US/Eastern'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None


MENUITEMS = [('Archive', 'archives.html'), ('About', 'pages/about.html'),('CV-Resume','pdfs/mohcine_madkour_cv.pdf'),]

STATIC_PATHS = ['images', 'pdfs']

COVER_IMG_URL = 'images/cover-img.jpg'


# Social widget
SOCIAL = (('linkedin', 'https://www.linkedin.com/in/mohcine-madkour-83a642b2'),
          ('github', 'https://github.com/mohcinemadkour/'),
          ('twitter', 'https://twitter.com/mohcinemadkour/'),)

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

#Set Disqus sitename
DISQUS_SITENAME = 'leafyleap-2'
#GOOGLE_ANALYTICS = "UA-52651211-1"
#ADDTHIS_PROFILE = "ra-54c667f5423e719f"
# Formatting for urls

ARTICLE_URL = "posts/{date:%Y}/{date:%m}/{slug}/"
ARTICLE_SAVE_AS = "posts/{date:%Y}/{date:%m}/{slug}/index.html"

CATEGORY_URL = "category/{slug}"
CATEGORY_SAVE_AS = "category/{slug}/index.html"

TAG_URL = "tag/{slug}/"
TAG_SAVE_AS = "tag/{slug}/index.html"

# Generate yearly archive
YEAR_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/index.html'

# Show most recent posts first
NEWEST_FIRST_ARCHIVES = False

# Static paths will be copied without parsing their contents
STATIC_PATHS = ['images', 'extra','pdfs']

# Ipython setting
NOTEBOOK_DIR = 'notebooks'
#EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')
IPYNB_STOP_SUMMARY_TAGS = [('div', ('class', 'input')), ('div', ('class', 'output')), 
						('div', ('content')), 'h1']


#MATH_JAX = {'color':'blue','align':'left'}
#MATH_JAX = {'tex_extensions': ['color.js','mhchem.js']}

# Shift the installed location of a file
#EXTRA_PATH_METADATA = {
#    'extra/CNAME': {'path': 'CNAME'},
#}

DEFAULT_PAGINATION = 10

# Show most recent posts first
NEWEST_FIRST_ARCHIVES = False

# Specify theme
THEME = "theme/pure"

# Plugins
PLUGIN_PATHS = ['./plugins']
#PLUGINS = ['gravatar', 'liquid_tags.youtube', 'liquid_tags.img', 'liquid_tags.notebook', 'pelican_gist']
MARKUP = ('md', 'ipynb')
#PLUGINS = ['gravatar', 'liquid_tags.youtube', 'liquid_tags.img', 'liquid_tags.include_code', 'liquid_tags.notebook', 'pelican_gist']
PLUGINS = ['render_math','gravatar', 'liquid_tags.youtube', 'liquid_tags.img',  'pelican_gist', 'ipynb.liquid', 'pelican_javascript']

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
MD_EXTENSIONS = ['codehilite(css_class=highlight)','extra']
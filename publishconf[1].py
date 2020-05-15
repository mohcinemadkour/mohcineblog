#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

#SITEURL = ''
RELATIVE_URLS = False

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

DELETE_OUTPUT_DIRECTORY = True

# Following items are often useful when publishing

#SITENAME = u'leafyleap'#-2.disqus.com'
#SITEURL = 'http://leafyleap.com'

#DISQUS_SITENAME = ""
#GOOGLE_ANALYTICS = ""
#SITEURL = 'https://www.leafyleap.com'
#DISQUS_SITENAME = 'leafleap'

#SITEURL = 'http://127.0.0.1:8000'
#DISQUS_SITENAME = 'leafyleap-2.disqus.com'

#leafyleap-2.disqus.com
#'http://www.leafyleap.com'

#SITEURL = 'http://leafyleap.com'
#SITENAME = 'leafyleap'#-2.disqus.com'

#SITEURL = 'http://leafyleap.com'
#SITENAME = 'leafyleap-2'#-2.disqus.com'
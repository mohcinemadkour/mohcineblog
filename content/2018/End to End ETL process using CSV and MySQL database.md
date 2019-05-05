Title: End to End ETL process using CSV files and MySQL database
Date: 2018-10-07 16:00
Category: Database 
Tags: MySQL, Database, Load data, CSV
Slug: Using only Python ETL a set of CSV files to a native MySQL database from end to end2
Author: Mohcine Madkour
Email: mohcine.madkour@gmail.com

This post explains an end to end process to move data from simple CSV files to a database server, in my case MySQL but you can do some tiny changes to have it work in any SQL language. I put the schema of the [database I generated from this code] (http://mohcinemadkour.github.io/DBSchema/) using schemaSPy. This is a 20 Gegabytes database I have cleaned and generated. I found this code also very useful wehn moving the database from development to production environments. I made this code in a modular format so the functions can be used if needed such as data_type function which detect the type of the column data and cast it to a python data type 


## Some notes

 Usually when I need to upload a CSV I will use Periscope
 Data's CSV functionality. It's fast, easy, allows me to join the data with all my databases, and automatically casts types and load the data. Sometimes, however, I like to interact directly with a MySQL cluster—usually for complex data transformations and modeling in Python. When interacting directly with a database, it can be a pain to write a create table statement and load your data. When the table is wide, you have two choices while writing your create table—spend the time to figure out the correct data types, or lazily import everything as text and deal with the type casting in SQL. The first is slow, and the second will get you in trouble down the road.

Here I show an example of this case when I upload 20 Gega of EHR data in a CSV format which are daunting 100+ columns wide. I wanted to load the data into MySQL server and rather than be generous in my data types, I wanted to use the proper columns. I decided to speed up the load process by writing a Python script, which turned into a fun exercise in data type detection and automated data loading to database.

## Check-list before start

First of all ... couple of things to check:

- Check your database is created and you have the required information (host name, database name, user, password)
- Names of columns can not have spaces
- Names of files will be the name of tables in the database
- The tables will be Droped if already exist

## Import libraries and provide path to data and information for connection

The first step is to load our data, import our libraries, and load the data into a CSV reader object. The csv library will be used to iterate over the data, and the ast library will be used to determine data type.

We will also use a few lists. "Longest" will be a list of the longest values in character length to specify varchar column capacity, "headers" will be a list of the column names, and "type_list" will be the updating list of column types as we iterate over our data.


```python
import mysql.connector
from mysql.connector import Error
import argparse
import csv
import sys
from os import listdir
from os.path import isfile, join
from itertools import count
import csv
import csv, ast 
#import psycopg2
from __future__ import print_function
```

Path and list of files I wanted to create the database from


```python
#mypath="/home/mohcine/Databox/Data/SortedData/left/new/"
mypath="Your path to where the CSV files are located"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.split(".")[1]=="csv"]
onlyfiles
```


Database info


```python
# Your creadential of the database
myhost=''
mydatabase=''
myuser=''
mypassword=''
```

## Find the Data Type

Once we have our data, we need to find the data type for each row. This means we need to evaluate every value and cast to the most restrictive option, from decimalized numbers to integers, and from integers to strings.

The function dataType does this. First, it evaluates to see if the value is text or a number, and then for the appropriate type of number if needed. This function consumes both the new data, and the current best type to evaluate against. 


```python
def dataType(val, current_type):
    try:
        # Evaluates numbers to an appropriate type, and strings an error
        t = ast.literal_eval(val)
    except ValueError:
        return 'varchar'
    except SyntaxError:
        return 'varchar'
    if type(t) in [int, long, float]:
        if (type(t) in [int, long]) and current_type not in ['float', 'varchar']:
            # Use smallest possible int type
            if (-32768 < t < 32767) and current_type not in ['int', 'bigint']:
                return 'smallint'
            elif (-2147483648 < t < 2147483647) and current_type not in ['bigint']:
                return 'int'
            else:
                return 'bigint'
            if type(t) is float and current_type not in ['varchar']:
                return 'decimal'
            else:
                return 'varchar'
```

## Create tables


```python
for f in onlyfiles:
    #Create table from the generated statement      
    fo = open(mypath+f, 'r')
    reader = csv.reader(fo)
    longest, headers, type_list = [], [], []
    # iterate over the rows in our CSV, call our function above, and populate our lists
    for row in reader:        
        if len(headers) == 0:
            headers = row
            for col in row:
                longest.append(0)
                type_list.append('')
        else:
            for i in range(len(row)):
                # NA is the csv null value
                if type_list[i] == 'varchar' or row[i] == 'NA':
                    pass
                else:
                    var_type = dataType(row[i], type_list[i])
                    type_list[i] = var_type
                if len(row[i]) > longest[i]:
                    longest[i] = len(row[i])
    fo.close()
    # And use our lists to write the SQL statement.
    
    statement = 'DROP TABLE IF EXISTS '+f.split(".")[0]
    try:
        conn = mysql.connector.connect(host=myhost,database=mydatabase,user=myuser,password=mypassword)
        if conn.is_connected(): 
            cursor = conn.cursor()
            cursor.execute(statement)
            conn.commit()
    except Error as e:
        print(e)
    finally:
        # Make sure data is committed to the database
        conn.commit()
        cursor.close()
        conn.close()

    statement = 'CREATE TABLE '+f.split(".")[0]+' ('
    for i in range(len(headers)):
        if type_list[i] == 'varchar':
            statement = (statement + '{} varchar({}),').format(headers[i].lower(), str(longest[i]))
        else:
            statement = (statement + '{} {}' + ',').format(headers[i].lower(), type_list[i])
    statement = statement[:-1] + ');'
    print (statement)
    #Create table from the generated statement  
    try:
        conn = mysql.connector.connect(host=myhost,database=mydatabase,user=myuser,password=mypassword)
        if conn.is_connected(): 
            print ("connected")
            cursor = conn.cursor()
            cursor.execute(statement)
            conn.commit()
    except Error as e:
        print(e)
    finally:
        # Make sure data is committed to the database
        conn.commit()
        cursor.close()
        conn.close()

```


## Load Data from local csv files


```python
for f in onlyfiles:
    #print f
    data=[]
    with open(mypath+f, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                cln = next(csv_reader)
                line_count += 1
            inserts=()
            for l in cln:
                inserts=inserts+(row[l],)
            data.append(inserts)
    #print len(cln)
    i=1
    col=""
    cal=""
    for l in cln:
        
        if i==1:
            col= "("+l+","
            val="(%s,"
            i=i+1
        elif (i < len(cln)):
            i=i+1
            col+= l+","
            val+="%s,"
        elif (i==len(cln)):
            col+=l+")"  
            val+="%s)"
            i=i+1
    try:
        conn = mysql.connector.connect(host=myhost,database=mydatabase,user=myuser,password=mypassword)
        if conn.is_connected():       
            query = "INSERT INTO "+f.split(".")[0]+col+"VALUES"+val
            cursor = conn.cursor()
            for each in data:
                cursor.execute(query,each)
                conn.commit()
    except Error as e:
        print(e)
    finally:
        # Make sure data is committed to the database
        conn.commit()
        cursor.close()
        conn.close()
```

## Check tables

Checking if the number of rows in the CVS files are matching the count of tables' rows in the database


```python
for f in onlyfiles:
    with open(mypath+f, 'r') as f1:
        csvlines = csv.reader(f1, delimiter=',')
        for lineNum, line in enumerate(csvlines):
            if lineNum == 0:
                c=len(line) 
    print "The csv file {} has {} rows and {} columns ".format(f,len(open(mypath+f).readlines()),c)
```



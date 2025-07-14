import sqlite3
import pandas as pd
conn= sqlite3.connect("instructor.db")

cursor_obj= conn.cursor()
cursor_obj.execute("Drop table  if exists instructor")
cursor_obj.execute("create table instructor(id integer Primary Key Not null, fname char(30), lname char(30), city char(30), ccode char(30))")
cursor_obj.execute("insert into instructor values(1,'tt', 'pp1', 'xx1','ca')")
cursor_obj.execute("insert into instructor values (2, 'Raul', 'Chong', 'Markham', 'CA'), (3, 'Hima', 'Vasudevan', 'Chicago', 'US')")
fetch= cursor_obj.execute('Select * from instructor')
panda_obj = pd.DataFrame(fetch.fetchall(), columns=['id','fname','lname','city','ccode'])
print(panda_obj)

obj = pd.read_sql('select * from instructor', conn)
print(obj)
if (obj.equals(panda_obj)):
    print('same')

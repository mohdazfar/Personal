import sqlite3

conn = sqlite3.connect('org.sqlite')
cur = conn.cursor()

cur.execute('''
DROP TABLE IF EXISTS Counts''')

cur.execute('''
CREATE TABLE Counts (org TEXT, count INTEGER)''')

fname = input('Enter file name: ')
if ( len(fname) < 1 ) : fname = 'mbox.txt'
fh = open(fname)
for line in fh:
    if not line.startswith('From: ') : continue
    pieces = line.split()
    email = pieces[1]
    orgPart = email.split("@")
    org = orgPart[1]
    breakLastEmail = org.split(".")
    orgName = breakLastEmail[0]
    #print (org)
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (orgName, ))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count) 
                VALUES ( ?, 1 )''', ( orgName, ) )
    else : 
        cur.execute('UPDATE Counts SET count=count+1 WHERE org = ?', 
            (orgName, ))
    # This statement commits outstanding changes to disk each 
    # time through the loop - the program can be made faster 
    # by moving the commit so it runs only after the loop completes
    conn.commit()

# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT DISTINCT org, count FROM Counts ORDER BY count DESC LIMIT 10'


print ("Counts:")
for row in cur.execute(sqlstr) :
    print (str(row[0]), row[1])

cur.close()

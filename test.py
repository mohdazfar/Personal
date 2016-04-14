import csv
import pandas as pd
import numpy as np

# list = [[302.26505463862571, 14.9436366221110735], [309.26505463862571, 4.9436366221110735]]
list = [[302, 14], [309, 4]]

a=[]
for i in list:
    a.append(i[0])
print(a)


#
# headers = ['Placement','Dataset','Edition','GlobalID','CodeGCR5','Series','SeriesUnindented','Attribute']
# countries = ['Albania','Algeria','Angola','Argentina','Armenia','Australia','Austria','Azerbaijan','Bahrain','Bangladesh','Barbados','Belgium','Belize','Benin','Bhutan','Bolivia','BosniaAndHerzegovina','Botswana','Brazil','BruneiDarussalam','Bulgaria','BurkinaFaso','Burundi','Cambodia','Cameroon','Canada','CapeVerde','Chad','Chile','China','Colombia','CostaRica','CôtedIvoire','Croatia','Cyprus','CzechRepublic','Denmark','DominicanRepublic','Ecuador','Egypt','ElSalvador','Estonia','Ethiopia','Finland','France','Gabon','Gambia','Georgia','Germany','Ghana','Greece','Guatemala','Guinea','Guyana','Haiti','Honduras','HongKong','Hungary','Iceland','India','Indonesia','Iran','Ireland','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Korea','Kuwait','KyrgyzRepublic','LaoPDR','Latvia','Lebanon','Lesotho','Liberia','Libya','Lithuania','Luxembourg','Macedonia','Madagascar','Malawi','Malaysia','Mali','Malta','Mauritania','Mauritius','Mexico','Moldova','Mongolia','Montenegro','Morocco','Mozambique','Myanmar','Namibia','Nepal','Netherlands','NewZealand','Nicaragua','Nigeria','Norway','Oman','Pakistan','Panama','Paraguay','Peru','Philippines','Poland','Portugal','PuertoRico','Qatar','Romania','Russia','Rwanda','SaudiArabia','Senegal','Serbia','Seychelles','SierraLeone','Singapore','SlovakRepublic','Slovenia','SouthAfrica','Spain','SriLanka','Suriname','Swaziland','Sweden','Switzerland','Syria','TaiwanChina','Tajikistan','Tanzania','Thailand','TimorLeste','TrinidadAndTobago','Tunisia','Turkey','Uganda','Ukraine','UnitedArabEmirates','UnitedKingdom','UnitedStates','Uruguay','Venezuela','Vietnam','Yemen','Zambia','Zimbabwe','AverageGCR','LatinAmerica','EmergingAndDevelopingAsia','MiddleEastNorthAfricaandPakistan','SubSaharanAfrica','CommonwealthOfIndependentStates','EmergingAndDevelopingEurope','Advancedeconomies','Lowincome','LowerMiddleincome','Uppermiddleincome','HighincomeOECD','HighincomenonOECD','ASEAN','Stage1','Transitionfrom1to2','Stage2','Transitionfrom2to3','Stage3']
#
# years = ['2015','2014','2013','2012','2011','2010','2009','2008','2007','2006']
#
#
# count = 0
# df = pd.read_csv('GCI.csv',encoding='latin-1')
# atribute = df[headers[7]]
#
# dataSet = df[headers[1]]
# edition = df[headers[2]]
# globalID = df[headers[3]]
# codeGCRS = df[headers[4]]
# series = df[headers[5]]
# seriesUnindented = df[headers[6]]
#
#
# counting = 0
# DataFrame = []
# subFrame = []
#
# for k in countries:
#     country = df[countries[count]]
#     valueHead = df[k]
#     count = count + 1
#     for j in range (0,len(atribute)):
#         if (atribute[j] == 'Value'):
#             #print(i, GlobalID[j], codeGCRS[j], series[j], seriesUnindented[j], k)
#             print(dataSet[j], edition[j],globalID[j], codeGCRS[j], series[j], seriesUnindented[j], k, valueHead[j])
#             x = [dataSet[j], edition[j],globalID[j], codeGCRS[j], series[j], seriesUnindented[j], k, valueHead[j]]
#             DataFrame.append(x)
# #DataFrame.append(subFrame)
#
# maindataFrame = np.array(DataFrame)
# print(maindataFrame)
# csvfile = pd.DataFrame(DataFrame,  columns= headers)
# csvfile.to_csv('DataFile.csv', sep=',', encoding='utf-8')
#
#



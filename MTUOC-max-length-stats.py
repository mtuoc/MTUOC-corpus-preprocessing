import codecs
import sys
import statistics

def calculate_stats(numbers):
    mean = statistics.mean(numbers)
    stdev = statistics.stdev(numbers)
    maximum = max(numbers)
    return mean, stdev, maximum
    
fentrada=sys.argv[1]
entrada=codecs.open(fentrada,"r",encoding="utf-8")

lengths=[]

for linia in entrada:
    linia=linia.rstrip()
    l=len(linia.split())
    lengths.append(l)
    
mean, stdev, maximum = calculate_stats(lengths)
print("Mean:", mean)
print("Standard Deviation:", stdev)
print("Maximum:", maximum)
m2sd=mean+2*stdev
print("M.+2*STD:", m2sd)


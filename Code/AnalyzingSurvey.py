import matplotlib.pyplot as plt
import os

# Script for generating plots from survey data

save_dir = 'D:/Dev/BA/jarlhengstmengel/Survey_Plots'

# Plotting number of analogies categorized as racist per demographic group
RasCount = [26, 27, 20, 24, 28, 26]
RasCountLabel =['Gesamt', 'Weiblich', 'Männlich', 'Rassismus-\nerfahrung', 'Nicht-Weiß', 'Weiß']
plt.bar(RasCountLabel, RasCount)
plt.ylabel('Anzahl als rassistisch bewertete Analogien')
plt.title('Anzahl rassistisch bewertete Analogien je demographische Gruppe')
plt.savefig(os.path.join(save_dir, 'PlotDemographischeGruppen.png'))
plt.clf()

# Plotting percent rating to nr of analogies of men and women
RasConfidenceWomen = [27, 25, 19, 15, 5]
RasConfidenceMen = [20, 19, 9, 2, 0]
Confidence = ['50%', '60%', '70%', '80%', '90%']
plt.plot(Confidence, RasConfidenceWomen, Confidence, RasConfidenceMen)
plt.ylabel('Anzahl als rassistisch bewertete Analogien')
plt.xlabel('Ab Prozentsatz')
plt.title('Unterschied der Stärke der Bewertung zwischen Frauen und Männern')
plt.savefig(os.path.join(save_dir, 'PlotFrauenVSMänner.png'))
plt.clf()

# Plotting number of analogies categorized as racist per age group
RasCountAge = [26, 25, 24]
AgeLabel = ['<30', '30-39', '>=40']
plt.bar(AgeLabel, RasCountAge)
plt.ylabel('Anzahl als rassistisch bewertete Analogien')
plt.xlabel('Alter in Jahren')
plt.title('Anzahl als rassistisch bewertete Analogie nach Altersgruppe')
plt.savefig(os.path.join(save_dir, 'PlotAltersgruppen.png'))
plt.clf()

# Plotting highest percent rating per analogy group
GroupMaxBlack = 0.87
GroupMaxJewish = 0.69
GroupMaxMuslim = 0.78
GroupMaxEasternEuropean = 0.85
GroupMaxRoma = 0.86
GroupMaxAsian = 0.83
GroupMaxLabel = ['Schwarz', 'Jüdisch', 'Muslimisch', 'Ost-\neuropäisch', 'Roma', 'Asiatisch']
plt.bar(GroupMaxLabel, (GroupMaxBlack, GroupMaxJewish, GroupMaxMuslim, GroupMaxEasternEuropean, GroupMaxRoma,
                        GroupMaxAsian))
plt.ylabel('Anteil Bewertung Analogie als rassistisch')
plt.title('Höchstbewerteten Analogien je Analogie-Gruppe')
plt.savefig(os.path.join(save_dir, 'PlotMaxAnalogiegruppen.png'))
plt.clf()

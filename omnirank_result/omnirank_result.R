library(RODBC)
library('RMySQL')
library(minerva)
library(ggplot2)
library(reshape2)
library(grid)
library(gridExtra)
setwd('/Users/wanghaiyang/workspace/R/ppd_omnirank')

#各个月的accuracy曲线
month <- 1:6
SVM <- c(0.715081967213,0.812786885246,0.829836065574,0.807540983607,0.804918032787,0.794098360656)
Logistic <- c(0.8,0.833770491803,0.840983606557,0.82262295082,0.82,0.804590163934)
RF <- c(0.797049180328,0.833442622951,0.850819672131,0.840327868852,0.840655737705,0.836721311475)
OMNIRank <- c(0.840327868852,0.844262295082,0.850819672131,0.840327868852,0.86,0.849180327869)
timeValue <- data.frame(month,SVM,Logistic,RF,OMNIRank)
timeValue_long <- melt(timeValue, id="month") # convert to long format
#cairo_ps('paper_data/paper_accuracy.eps')
ggplot(data=timeValue_long)+
  geom_histogram(aes(x=month, y=value, fill=as.factor(variable)), position = "dodge", stat="identity", alpha = 1,  width = 0.8)+
  theme_bw()+
  theme(text=element_text(family = "Microsoft YaHei",face="bold",size=25),legend.title=element_blank(), axis.text.x=element_text(angle=0, color = "black", size=20), axis.text.y=element_text(color = "black", size=20), plot.title=element_blank(), plot.margin = unit(c(1.5,0.7,0.7,0.5), "cm"), axis.title.x=element_text(vjust=-1), axis.title.y=element_text(vjust=1.5))+ 
  labs(x='month',y='accuracy',title='OMNIRank评分准确率与AUC')+
  scale_x_continuous(breaks=c(1,2,3,4,5,6), labels=c("2015-11", "2015-12", "2016-01", "2016-02", "2016-03", "2016-04"))+
  scale_color_brewer(palette="Set3")+
  scale_fill_brewer(palette="Set3")+
  coord_cartesian(ylim=c(0.7, 0.87))
#dev.off()

#各个月的auc曲线
month <- 1:6
SVM <- c(0.729256924713,0.748863338106,0.77632426896,0.797578824513,0.81327557781,0.816653787828)
Logistic <- c(0.805231948091,0.814425451392,0.82320828334,0.820215115951,0.827436627889,0.825864402262)
RF <- c(0.852796686843,0.867216057222,0.870548980362,0.865983488418,0.87876480364,0.8821367606)
OMNIRank = c(0.852089625538,0.858188191289,0.869886839283,0.882064123965,0.899087775476,0.905489157118)
timeValue <- data.frame(month,SVM,Logistic,RF,OMNIRank)
timeValue_long <- melt(timeValue, id="month") # convert to long format
ggplot(data=timeValue_long)+
  geom_histogram(aes(x=month, y=value, fill=as.factor(variable)), position = "dodge", stat="identity", alpha = 1,  width = 0.8)+
  theme_bw()+
  theme(text=element_text(family = "Microsoft YaHei",face="bold",size=25),legend.title=element_blank(), axis.text.x=element_text(angle=0, color = "black", size=20), axis.text.y=element_text(color = "black", size=20), plot.title=element_blank(), plot.margin = unit(c(1.5,0.7,0.7,0.5), "cm"), axis.title.x=element_text(vjust=-1), axis.title.y=element_text(vjust=1.5))+ 
  labs(x='month',y='auc',title='OMNIRank评分准确率与AUC')+
  scale_x_continuous(breaks=c(1,2,3,4,5,6), labels=c("2015-11", "2015-12", "2016-01", "2016-02", "2016-03", "2016-04"))+
  scale_color_brewer(palette="Set3")+
  scale_fill_brewer(palette="Set3")+
  coord_cartesian(ylim=c(0.7, 0.92))


#各模型评分分布图
data_source <- c('paper_data/dl_201504.csv', 'paper_data/logistic_201504.csv', 'paper_data/rf_201504.csv', 'paper_data/svm_201504.csv')
model_type <- c('OMNIRank', 'Logistic', 'RandomForest', 'SVM')

ppdFile <- data_source[4]
options(digits=15)
ppdData <- read.table(file=ppdFile, 
                      head=T, 
                      sep=',', 
                      col.names = c('label','name','score'),
                      nrows = -1)

ppdData$label[ppdData$label==0] <- 'normal'
ppdData$label[ppdData$label==1] <- 'question'
#频率直方图
p4 <- ggplot(ppdData)+geom_histogram(aes(x=score, fill=as.factor(label)), position = "dodge", 
                                       alpha = 0.8, binwidth=0.03)+
  labs(x='score',y='amount',title=model_type[4])+
  theme_bw()+
  #theme(text=element_text(family = "Microsoft YaHei",face="bold",size=25),legend.title=element_blank(), axis.title.x=element_text(vjust=-1), axis.title.y=element_text(vjust=1.5), plot.title=element_text(vjust=2.5, size=28), axis.text.x=element_text(angle=0, color = "black", size=25),legend.position="right", axis.text.y=element_text(color = "black", size=25), plot.margin = unit(c(1.5,0.7,0.7,0.5), "cm"))+
  theme(legend.position="none", text=element_text(family = "Microsoft YaHei",face="bold",size=25),legend.title=element_blank(), axis.title.x=element_text(vjust=-1), axis.title.y=element_text(vjust=1.5), plot.title=element_text(vjust=2.5, size=25), axis.text.x=element_text(angle=0, color = "black", size=20),legend.position="right", axis.text.y=element_text(color = "black", size=20), plot.margin = unit(c(1.5,0.7,0.7,0.5), "cm"))+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  scale_x_continuous(breaks=c(0,0.2,0.4,0.6,0.8,1.0), labels=c(0,0.2,0.4,0.6,0.8,1.0))

gp1<- ggplot_gtable(ggplot_build(p1))
gp2<- ggplot_gtable(ggplot_build(p2))
gp3<- ggplot_gtable(ggplot_build(p3))
gp4<- ggplot_gtable(ggplot_build(p4))
maxWidth = unit.pmax(gp1widths[2:3],gp2widths[2:3],gp2widths[2:3], gp3widths[2:3],gp4widths[2:3],gp4widths[2:3])
gp1$widths[2:3] <- maxWidth
gp2$widths[2:3] <- maxWidth
gp3$widths[2:3] <- maxWidth
gp4$widths[2:3] <- maxWidth
grid.arrange(gp1, gp2, gp3, gp4)#, nrow=4设置排列方式


#平台分类统计饼图
dt <- data.frame(A = c(1672, 240, 383, 737, 18), B = c('Normal','Closed Down','Withdraw Failure','Runaway Bosses','Investigation Involvements'))
dt = dt[order(dt$A, decreasing = TRUE),]
myLabel = as.vector(dt$B)
#myLabel = paste(myLabel, "(", round(dt$A / sum(dt$A) * 100, 2), "%)", sep = "")
myLabel = paste( round(dt$A / sum(dt$A) * 100, 2), "%" )
p <- ggplot(dt, aes(x = "", y = A, fill = B))+
  theme_bw()+
  geom_bar(stat = "identity", width = 1)+
  coord_polar(theta = "y")+
  labs(x = "", y = "", title = "")+
  theme(axis.ticks = element_blank())+
  theme(legend.title = element_blank(), legend.position = "right")+
  #scale_fill_discrete(breaks = dt$B, labels = myLabel)+
  theme(axis.text.x = element_blank())+
  #geom_text(aes(y = A/2 + c(0, cumsum(A)[-length(A)]), x = sum(A)/2500, label = myLabel), size = 5)+## 在图中加上百分比：x 调节标签到圆心的距离, y 调节标签的左右位置
  scale_color_brewer(palette="Set3")+
  scale_fill_brewer(palette="Set3")
p
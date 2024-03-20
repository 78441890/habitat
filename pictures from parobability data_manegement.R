## 基于概率阈值绘制ROC曲线、DCA曲线和校准曲线

setwd("E:\\UAI_Program\\2-ZhongshanHospital\\10-wangdacheng\\2-瘤周亚区分析的提取特征结果\\7-Picyures\\RCOS_Dilation")

# library packages
library(ROCR);library(pROC);library(stringr)
library(ggsci);library(ggplot2)
library(rmda)
library(reshape2)
library(RColorBrewer)
library(ResourceSelection) 
library(reportROC)


## functions

ROCSFun <- function(Picdata,TarCol,OutName,colorm){
	Group <- unique(Picdata[,2])
	for(i in 1:length(Group)){
		Data1 <- Picdata[which(Picdata[,2]==Group[i]),TarCol]
		names(Data1)[-1] <- names(Data1)[-1]
		# 分别将三个模型的FPR和TPR计算出来，保存，用于后续ggplot2作图
		Fpr <- list();Tpr <- list();AUC <- c()
		for(j in 2:ncol(Data1)){
			Data2 <- Data1[,c(1,j)]
			ROC <- roc(Data2[,1],Data2[,2],ci=T)
			Perf <- performance(prediction(Data2[,2],Data2[,1]),"tpr","fpr")
			Fpr[[j-1]] <- Perf@x.values[[1]];Tpr[[j-1]] <- Perf@y.values[[1]]	

			AUC <- c(AUC,round(ROC$auc,3))
		}
		# ROC曲线准备
		ModelName <- c(); TPR <- c(); FPR <- c()
		for(j in 1:length(Tpr)){
			ModelName <- c(ModelName,rep(paste0(names(Data1)[j+1]," model AUC=",AUC[j]),length(Tpr[[j]])))
			if(j >= 1){
				TPR <- c(TPR,Tpr[[j]])
				FPR <- c(FPR,Fpr[[j]])
			}
			else{
				TPR <- c(TPR,Fpr[[j]])
				FPR <- c(FPR,Tpr[[j]])
			}

		}
		GGROCData <- data.frame(ModelName,TPR ,FPR)
		GGROCData[,1] <- factor(GGROCData[,1],levels=unique(GGROCData[,1]))
	
		# ROC曲线制作
		Pic <- ggplot(aes(x=FPR,y=TPR,color=ModelName),data=GGROCData)+
			geom_line(lwd=0.8)+
			#scale_color_lancet()+ # npg可以换成nejm,lancet,jama,jco,ucscgb,d3,locuszoom,tron,futurama
			scale_color_manual(values=colorm[1:7])+
			theme(legend.position = c(0.67,0.2),legend.key.height=unit(1.5,"line"),
				axis.title.x =element_text(size=14), 
				axis.title.y=element_text(size=14),axis.text=element_text(size=14,color="black"),
				#axis.text.x = element_text(angle=90,vjust=.53),
				panel.grid.major=element_line(color="white"),
				panel.grid.minor=element_line(color="white"),
				panel.background=element_rect(fill ="white",color="black"),
				legend.key=element_rect(fill ="white",color="white"),
				legend.text=element_text(size=14), # 图例标签调整
				legend.title=element_blank())+ # 隐藏图例的标题
			#scale_color_discrete(breaks=unique(GGROCData[,1]),labels=paste0(names(Data1)[-1]," model AUC=",AUC))+ # 修改图例标签
			geom_abline(intercept=0)+
			labs(x="False positive rate",y="True positive rate")
		## 保存图片
		pdf(paste0("3-ROC curve in ",OutName ,"-",Group[i],".pdf"),family="Times",height=6,width=8)
		print(Pic)
		dev.off()

		# delong test
		# 保存每个模型的ROC结果
		ROC <- list()
		for(j in 2:ncol(Data1)){
			Data2 <- Data1[,c(1,j)]
			ROC[[j-1]] <- roc(Data2[,1],Data2[,2])	
		}
		# 利用ROC结果进行Delong检验
		DelongP <- c()
		for(m in 1:length(ROC)){
			for(n in 1:length(ROC)){
				DelongP <- c(DelongP,roc.test(ROC[[m]],ROC[[n]],method="delong")$p.value)
			} 
		}
		DelongP <- as.data.frame(matrix(DelongP,ncol=length(ROC),byrow=T))
		DelongP <- round(DelongP ,3)
		names(DelongP) <- names(Data1)[-1]
		row.names(DelongP) <- names(Data1)[-1]
		write.csv(DelongP,paste0("4-Delong_result_",OutName ,Group[i],".csv"))
	}

}


## Decision curve analysis
DCAFun <- function(Picdata,TarCol,OutName,colorm,LegendLoc,Width){
	Group <- unique(Picdata[,2])
	for(i in 1:length(Group)){#
		DCAData2 <- Picdata[Picdata[,2]==Group[i],TarCol]
		# 决策曲线
		DCAPic <- list()
		for(k in 2:ncol(DCAData2)){
			Formular <- as.formula(paste0("Label~",names(DCAData2)[k]))
			DCATmp <- decision_curve(Formular,data=DCAData2 ,family = binomial(link ='logit'),
				thresholds= seq(0,1,by = 0.01),study.design = 'cohort',policy = "opt-in",bootstraps = 50)
			DCAPic[[k-1]] <- DCATmp
		}
		names(DCAPic) <- names(DCAData2)[-1]


		## 用ggplot来画图
		l <- length(DCAPic[[1]][["derived.data"]][["NB"]])/3
		#设置一个NA数据框用来循环取数据
		dc <- data.frame('NBl'=rep(NA,l),
                 'NB'=rep(NA,l),
                 'NBu'=rep(NA,l),
                 'HRT'=rep(NA,l),
                 'CBR'=rep(NA,l),
                 'Model'=rep(NA,l))

		#建立空数据框用来合并数据
		dat <- data.frame()
		for (j in names(DCAPic)) {
 			 dc$NBl <- DCAPic[[j]][["derived.data"]][["NB_lower"]][1:l]
 			 dc$NB<- DCAPic[[j]][["derived.data"]][["NB"]][1:l]
 			 dc$NBu <- DCAPic[[j]][["derived.data"]][["NB_upper"]][1:l]
 			 dc$HRT <- DCAPic[[j]][["derived.data"]][["thresholds"]][1:l]
			 dc$CBR <- DCAPic[[j]][["derived.data"]][["cost.benefit.ratio"]][1:l]
  			 dc$Model <- j
  			 #按行合并
 			 dat <- rbind(dat,dc)
 		 }
		#正经模型的数据取完了，现在取All和None的
		dc$NBl <- DCAPic[[1]][["derived.data"]][["NB_lower"]][(2*l+1):(3*l)]
		dc$NB<- DCAPic[[1]][["derived.data"]][["NB"]][(2*l+1):(3*l)]
		dc$NBu <- DCAPic[[1]][["derived.data"]][["NB_upper"]][(2*l+1):(3*l)]
		dc$HRT <- DCAPic[[1]][["derived.data"]][["thresholds"]][(2*l+1):(3*l)]
		dc$CBR <- DCAPic[[1]][["derived.data"]][["cost.benefit.ratio"]][(2*l+1):(3*l)]
		dc$Model <- rep('None',l)
		dat <- rbind(dat,dc)
		#dat[which(is.na(dat$NBl)),'CBR'] <- '100:1'#这一行画图没用到，纯粹强迫症
		dat[which(is.na(dat$NBl)),1:3] <- dat[which(is.na(dat$NBl))-1,1:3]
		unique(dat[,6])
		#加入all
		dc$NBl <- DCAPic[[1]][["derived.data"]][["NB_lower"]][(l+1):(2*l)]
		dc$NB<- DCAPic[[1]][["derived.data"]][["NB"]][(l+1):(2*l)]
		dc$NBu <- DCAPic[[1]][["derived.data"]][["NB_upper"]][(l+1):(2*l)]
		dc$HRT <- DCAPic[[1]][["derived.data"]][["thresholds"]][(l+1):(2*l)]
		dc$CBR <- DCAPic[[1]][["derived.data"]][["cost.benefit.ratio"]][(l+1):(2*l)]
		dc$Model <- rep('All',l)
		dat <- rbind(dat,dc)
		#最后一行含NA值
		dat <- dat[-nrow(dat),]
		#去除负值较大的行(主要是all那个模型)，咱们画图y轴-0.1就差不多得了
		dat <- dat[dat$NB>-0.1,]
		dat$Model <- factor(dat$Model,levels = rev(unique(dat$Model)))
		#找一个y轴上限
		lmax <- max(dat$NB)
		#colorm <- c('grey','black',"red","blue","green")#brewer.pal(length(unique(dat$Model))-2,'Dark2')
		#Cost:Benefit Ratio标签（这里基本上可以不变，因为x轴就那么长，没必要细化了）
		cb <- data.frame(x=c(1/101,0.2,0.4,0.6,0.8,100/101),lab=c('1:100','1:4','2:3','3:2','4:1','100:1'))
		#假x轴的标签（实际是画的一条线）
		tik <- data.frame(x=c(0,0.2,0.4,0.6,0.8,1),
                  y=rep(-0.1,6))

		#  不要 all和none
		DatTmp <- dat
		DatTmp[,6] <- factor(DatTmp[,6],levels=unique(DatTmp[,6]))

		DCAPlot <- ggplot()+
		  geom_line(data=DatTmp,lwd=0.8,aes(x=HRT,y=NB,color=Model))+ 
		  scale_color_manual(values=colorm)+
		  scale_y_continuous(name = 'Net Benefit',breaks = seq(0,max(DatTmp$NB),by=0.1),limits = c(-0.18,max(DatTmp$NB)+0.1),expand = c(0,0))+
		  scale_x_continuous(name = 'Cost:Benefit Ratio',breaks =cb$x,expand = c(0.01,0),labels = cb$lab)+
		  geom_segment(aes(x=0,y=-0.1,xend=1,yend=-0.1),color='black',size=0.5)+
		  geom_segment(data=tik,aes(x=x,y=y,xend=x,yend=y-0.01))+
		  geom_text(tik,mapping = aes(x=x,y=-0.1,label=x),vjust=2.5,size=5)+
		  geom_text(data=NULL,aes(x=0.5,y=-0.1,label='High Risk Threshold'),colour = 'black',size=5,vjust=4)+
		  guides(color=guide_legend(ncol=1, byrow=TRUE))+
		  theme_bw()+
		  theme(panel.grid.minor = element_line(colour = NA),
		        legend.position=LegendLoc,legend.text=element_text(size=14),legend.key.height=unit(1.5,"line"),
		        legend.background = element_blank(),legend.title=element_blank(),
		        #这个边框如果调整不好的话建议去除
		        legend.box.background = element_blank(),
 		       panel.border = element_rect(),
 		       axis.title = element_text(size=14,colour = 'black'),
		        axis.text = element_text(size = 14,colour = 'black'))+
		  guides(color=guide_legend(nrow=3, byrow=TRUE))
		  
		  #scale_color_lancet()+ # npg可以换成nejm,lancet,jama,jco,ucscgb,d3,locuszoom,tron,futurama


		pdf(paste0("5-DCA curve in ",OutName ,Group[i],".pdf"),family="Times",height=6,width=Width)
			print(DCAPlot)
		dev.off() 
	}
}
## 矫正曲线，分成4组，计算每一组的实际概率和预测概率
CalibrationFun <- function(Picdata,TarCol,Thres,OutName,colorm){
	Group <- unique(Picdata[,2])
	for(i in 1:length(Group)){
		Data1 <- Picdata[which(Picdata[,2]==Group[i]),TarCol]
		names(Data1)[-1] <- names(Data1)[-1]
		# 分组
		Diff <- round(nrow(Data1)/Thres,0)
		TruProb <- c();PreProb <- c();Model <- c()
		for(j in 2:ncol(Data1)){
			tmpdata <- Data1[,c(1,j)]
			tmpdata <- tmpdata[order(tmpdata[,2],decreasing=F),]
			for(k in 1:3){
				needrows <- (Diff*k-(Diff-1)):(Diff*k)
				TruProb <- c(TruProb,mean(tmpdata[needrows,1]))
				PreProb <- c(PreProb,mean(tmpdata[needrows,2]))
			}
			Model <- c(Model,rep(names(Data1)[j],3))
		}
		DatTmp <- data.frame(Model,TruProb,PreProb)
		DatTmp[,1] <- paste0(DatTmp[,1]," model")
		DatTmp[,1] <- factor(DatTmp[,1],levels=unique(DatTmp[,1]))
		DCAPlot <- ggplot()+
		  geom_line(data=DatTmp,lwd=0.8,aes(x=TruProb,y=PreProb,color=Model))+ 
		  geom_point(data=DatTmp,size=2,aes(x=TruProb,y=PreProb,color=Model))+
		  geom_abline(intercept=0,slope=1,color="black",linetype="dashed",lwd=0.7)+
		  scale_color_manual(values=colorm[1:7])+
		  scale_y_continuous(name = 'Mean predicted probability',breaks = seq(0,1,by=0.2),limits = c(0,1),expand = c(0,0))+
		  scale_x_continuous(name = 'Fraction of positives',breaks =seq(0,1,by=0.2),limits = c(0,1),expand = c(0,0))+
		  theme_bw()+
		  theme(legend.position = c(0.75,0.2),legend.key.height=unit(1.5,"line"),
			axis.title.x =element_text(size=12), 
			axis.title.y=element_text(size=12),axis.text=element_text(size=12,color="black"),
			#axis.text.x = element_text(angle=90,vjust=.53),
			panel.grid.major=element_line(color="white"),
			panel.grid.minor=element_line(color="white"),
			panel.background=element_rect(fill ="white",color="black"),
			legend.key=element_rect(fill ="white",color="white"),
			legend.text=element_text(size=12), # 图例标签调整
			legend.title=element_blank())#+  隐藏图例的标题
			#geom_abline(intercept=0))
		## 保存图片
		pdf(paste0("6-calibration curve in ",OutName,Group[i],".pdf"),family="Times",height=5,width=7)
			print(DCAPlot)
		dev.off()
	}
}

## hosmer lemshow test and brier score
HLBerierCalc <- function(Picdata,TarCol,Threshld){
	Group <- unique(Picdata[,2])
	HRPvalue <- c()
	for(i in 1:length(Group)){
		Data1 <- Picdata[which(Picdata[,2]==Group[i]),TarCol]
		names(Data1)[-1] <- names(Data1)[-1]
		for(j in 2:ncol(Data1)){
			hl <- hoslem.test(Data1[,1], Data1[,j], g=6)
			HRPvalue <- c(HRPvalue ,round(hl$p.value,3))
		}
	}
	HRPvalueData <- matrix(HRPvalue,ncol=2,byrow=F) %>% as.data.frame()
	row.names(HRPvalueData) <- names(Data1)[-1]
	names(HRPvalueData) <- Group
	write.csv(HRPvalueData,"7-HRresult.csv")

	# berier score
	BSvalue <- c()
	for(i in 1:length(Group)){
		Data1 <- Picdata[which(Picdata[,2]==Group[i]),TarCol]
		Events <- unique(Data1[,1])
		for(j in 2:ncol(Data1)){
			Label <- Data1[,1]
			Prediction <- ifelse(Data1[,j]>Threshld,1,0)
			BSTmp <- c()
			for(k in 1:length(Events)){
				Locate <- which(Label==Events[k])
				PreLocate <- which(Prediction==Events[k])
				Ot <- length(Locate)/length(Label)
				Ft <- length(PreLocate)/length(Label)
				Res <- (Ft-Ot)^2
				BSTmp <- c(BSTmp,Res)
			}
			BSvalue <- c(BSvalue,sum(BSTmp)/2)
		}
	}
	BSData <- matrix(BSvalue,ncol=2,byrow=F) %>% as.data.frame()
	row.names(BSData) <- names(Data1)[-1]
	names(BSData ) <- Group
	write.csv(BSData ,"7-BSData.csv")
}


## 约登指数
YouDenCalcu <- function(Picdata,TarCol){
	Group <- unique(Picdata[,2])
	Cutoff <- c()
	for(i in 2:length(Group)){
		Data1 <- Picdata[which(Picdata[,2]==Group[2]),TarCol]
		names(Data1)[-1] <- names(Data1)[-1]
		for(j in 2:ncol(Data1)){
			rocs <- roc(Data1[,1],Data1[,j],ci=T)
			Perf <- performance(prediction(Data1[,j],Data1[,1]),"tpr","fpr")
			Fpr <- Perf@x.values[[1]];Tpr <- Perf@y.values[[1]]
			Youdentmp <- Tpr - Fpr
			Cutoff <- c(Cutoff,Youdentmp[which.max(Youdentmp )])
		}
	}
	# 各评价指标计算
	Auc <- c();Acc <- c();Sen <- c();Spe <- c();Precision <- c()
	for(i in 1:length(Group)){
		ROCData <- Picdata[which(Picdata[,2]==Group[i]),TarCol]
		for(j in 2:ncol(ROCData)){
			rocobj <- roc(ROCData[,1], ROCData[,j],ci=T)
			Label <- ROCData[,1]
			Prediction1 <- ifelse(ROCData[,j]>Cutoff[j-1],1,0)
			Prediction2 <- ifelse(ROCData[,j]>0.5,1,0)
			if(unique(Prediction1)<=2){
				Prediction <- Prediction2
			}
			else{
				Prediction <- Prediction1
			}
			TableData <- table(Tru=Label,pre=Prediction)
	
			Auc <- c(Auc,paste0(round(as.numeric(rocobj$auc),3),"(",round(rocobj$ci[1],3),"-",round(rocobj$ci[3],3),")"))
			Acc <- c(Acc,round(sum(TableData[1,1],TableData[2,2])/sum(TableData),3))
			Sen <- c(Sen,round(sum(TableData[2,2])/sum(TableData[2,]),3))
			Spe <- c(Spe,round(sum(TableData[1,1])/sum(TableData[1,]),3))
			Precision <- c(Precision,round(sum(TableData[2,2])/sum(TableData[,2]),3))
		}
		F1Score <- 2*Precision*Sen/c(Precision+Sen)
		TestDataPerfomance <- data.frame(Model=names(ROCData)[-1],Auc,Acc,Sen,Spe,Precision,F1Score)
		write.csv(TestDataPerfomance ,paste0('3-Perfomance-',Group[i],'.csv'))
	}
}


## 通过reportROC包确定分类阈值
ReportROC <- function(Picdata,TarCol){
	Group <- unique(Picdata[,2])
	# 分类阈值确定与评价指标计算
	Cutoff <- c();Auc <- c();Acc <- c();Sen <- c();Spe <- c();Precision <- c()
	Data1 <- Picdata[which(Picdata[,2]=="Train"),TarCol]
	for(j in 2:ncol(Data1)){
		# report ROC
		res.num<- reportROC(gold=Data1[,1],#金标准
       		  predictor=Data1[,j],#预测变量
        		  important="se",#  选择截断值时候哪个更重要
       		  plot=F)
		res.num
		Cutoff <- c(Cutoff,res.num[1] %>% as.numeric())
		Auc <- c(Auc,paste0(as.numeric(res.num[2]),"(",as.numeric(res.num[4]),"-",as.numeric(res.num[5]),")"))
		Acc <- c(Acc,as.numeric(res.num[7]))
		Sen <- c(Sen,as.numeric(res.num[10]))
		Spe <- c(Spe,as.numeric(res.num[13]))
		Precision <- c(Precision,as.numeric(res.num[22]))
	}
	F1Score <- round(2*Precision*Sen/c(Precision+Sen),3)
	DataPerfomance <- data.frame(Model=names(Data1)[-1],Auc,Acc,Sen,Spe,Precision,F1Score)
	write.csv(DataPerfomance,paste0('3-Perfomance-',"Train",'.csv'))

	for(i in setdiff(Group,"Train")){
		ROCData <- Picdata[which(Picdata[,2]==i),TarCol]
		Auc <- c();Acc <- c();Sen <- c();Spe <- c();Precision <- c()
		for(j in 2:ncol(ROCData)){
			rocobj <- roc(ROCData[,1], ROCData[,j],ci=T)
			Label <- ROCData[,1]
			Prediction1 <- ifelse(ROCData[,j]>Cutoff[j-1],1,0)
			Prediction2 <- ifelse(ROCData[,j]>0.5,1,0)
			if(length(unique(Prediction1))<=2){
				Prediction <- Prediction2
			}
			else{
				Prediction <- Prediction1
			}
			TableData <- table(Tru=Label,pre=Prediction)
	
			Auc <- c(Auc,paste0(round(as.numeric(rocobj$auc),3),"(",round(rocobj$ci[1],3),"-",round(rocobj$ci[3],3),")"))
			Acc <- c(Acc,round(sum(TableData[1,1],TableData[2,2])/sum(TableData),3))
			Sen <- c(Sen,round(sum(TableData[2,2])/sum(TableData[2,]),3))
			Spe <- c(Spe,round(sum(TableData[1,1])/sum(TableData[1,]),3))
			Precision <- c(Precision,round(sum(TableData[2,2])/sum(TableData[,2]),3))
		}
		F1Score <- round(2*Precision*Sen/c(Precision+Sen),3)
		TestDataPerfomance <- data.frame(Model=names(ROCData)[-1],Auc,Acc,Sen,Spe,Precision,F1Score)
		write.csv(TestDataPerfomance ,paste0('3-Perfomance-',i,'.csv'))
	}
}

## 通过reportROC包各自确定数据集的分类阈值，获得更好结果
ReportROC2 <- function(Picdata,TarCol){
	Group <- unique(Picdata[,2])
	for(i in Group){
		ROCData <- Picdata[which(Picdata[,2]==i),TarCol]
		Cutoff <- c();Auc <- c();Acc <- c();Sen <- c();Spe <- c();Precision <- c()
		for(j in 2:ncol(ROCData)){
			# report ROC
			res.num<- reportROC(gold=ROCData[,1],#金标准
       		  predictor=ROCData[,j],#预测变量
        		  important="se",#  选择截断值时候哪个更重要
       		  plot=F)
			res.num
			Cutoff <- c(Cutoff,res.num[1] %>% as.numeric())
			Auc <- c(Auc,paste0(as.numeric(res.num[2]),"(",as.numeric(res.num[4]),"-",as.numeric(res.num[5]),")"))
			Acc <- c(Acc,as.numeric(res.num[7]))
			Sen <- c(Sen,as.numeric(res.num[10]))
			Spe <- c(Spe,as.numeric(res.num[13]))
			Precision <- c(Precision,as.numeric(res.num[22]))
		}
		F1Score <- round(2*Precision*Sen/c(Precision+Sen),3)
		TestDataPerfomance <- data.frame(Model=names(ROCData)[-1],Cutoff,Auc,Acc,Sen,Spe,Precision,F1Score)
		write.csv(TestDataPerfomance ,paste0('3-Perfomance-',i,'.csv'))
	}
}


### load data
Picdata <- read.csv("1_HCCDilation_Probability2.csv")#[,-2]
#names(Picdata)[c(6:8,12:14)] <- c("Pre_10mm","Pre_10mm_cluster1","Pre_10mm_cluster2",
#	"Pre_5mm","Pre_5mm_cluster1","Pre_5mm_cluster2")
str(Picdata)
#Picdata <- Picdata[,c(1:5,12:14,9:11,6:8)]
# ROC function apply
TarColLr1 <- c(1,3,6,9,12)
TarColLr2 <- c(1,3+1,6+1,9+1,12+1)
TarColLr3 <- c(1,3+2,6+2,9+2,12+2)
colorm <- c(rainbow(length(TarColLr1)-1),"black","grey")

## ROC曲线 保存
ROCSFun(Picdata,TarColLr1,"DIlation",colorm)
ROCSFun(Picdata,TarColLr2,"DIlation_cluster1",colorm)
ROCSFun(Picdata,TarColLr3,"DIlation_cluster2",colorm)


## DCA 曲线保存
DCAData <- Picdata
names(DCAData)[TarColLr2] <- names(DCAData)[TarColLr1]
names(DCAData)[TarColrf2] <- names(DCAData)[TarColrf1]
names(DCAData)[TarColsvm2] <- names(DCAData)[TarColsvm1]

DCAFun(DCAData,TarColLr1,"tmp",colorm,"bottom",6)
DCAFun(DCAData,TarColLr2,"LR-combine",colorm,"right",9)
DCAFun(DCAData,TarColrf1 ,"RF-single",colorm,"right",8)
DCAFun(DCAData,TarColrf2 ,"RF-combine",colorm,"right",9)
DCAFun(DCAData,TarColsvm1 ,"SVM-single",colorm,"right",8)
DCAFun(DCAData,TarColsvm2 ,"SVM-combine",colorm,"right",9)
DCAFun(Picdata,TarColsvm2 ,"SVM-combine_trueLegend",colorm,"right",9)

## 校准曲线保存
CalibrationFun(Picdata,TarColLr1,5,"tmp",colorm)
CalibrationFun(Picdata,TarColLr2,"LR-combine",colorm)
CalibrationFun(Picdata,TarColrf1 ,"RF-single",colorm)
CalibrationFun(Picdata,TarColrf2 ,"RF-combine",colorm)
CalibrationFun(Picdata,TarColsvm1 ,"SVM-single",colorm)
CalibrationFun(Picdata,TarColsvm2 ,"SVM-combine",colorm)

## HL test + berier score
HLBerierCalc(Picdata,c(1,3:ncol(Picdata)),0.5)

## 通过约登指数计算评价指标
YouDenCalcu(Picdata,c(1,3:ncol(Picdata)))

## 通过reportROC包计算评价指标
ReportROC(Picdata,c(1,3:ncol(Picdata)))
ReportROC2(Picdata,c(1,3:ncol(Picdata)))


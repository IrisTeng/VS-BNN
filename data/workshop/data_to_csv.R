################################################
###  Taken from `BKMR code - NHANES - final.R`
################################################


## load required libraries 
library(bkmr)
library(ggplot2)


################################################
###         Data Manipulation                ###
################################################


## read in data and only consider complete data 
## this drops 327 individuals, but BKMR does not handle missing data
nhanes <- na.omit(read.csv("studypop.csv"))

## center/scale continous covariates and create indicators for categorical covariates
nhanes$age_z         <- scale(nhanes$age_cent)         ## center and scale age
nhanes$agez_sq       <- nhanes$age_z^2                 ## square this age variable
nhanes$bmicat2       <- as.numeric(nhanes$bmi_cat3==2) ## 25 <= BMI < 30
nhanes$bmicat3       <- as.numeric(nhanes$bmi_cat3==3) ## BMI >= 30 (BMI < 25 is the reference)
nhanes$educat1       <- as.numeric(nhanes$edu_cat==1)  ## no high school diploma
nhanes$educat3       <- as.numeric(nhanes$edu_cat==3)  ## some college or AA degree
nhanes$educat4       <- as.numeric(nhanes$edu_cat==4)  ## college grad or above (reference is high schol grad/GED or equivalent)
nhanes$otherhispanic <- as.numeric(nhanes$race_cat==1) ## other Hispanic or other race - including multi-racial
nhanes$mexamerican   <- as.numeric(nhanes$race_cat==2) ## Mexican American 
nhanes$black         <- as.numeric(nhanes$race_cat==3) ## non-Hispanic Black (non-Hispanic White as reference group)
nhanes$wbcc_z        <- scale(nhanes$LBXWBCSI)
nhanes$lymphocytes_z <- scale(nhanes$LBXLYPCT)
nhanes$monocytes_z   <- scale(nhanes$LBXMOPCT)
nhanes$neutrophils_z <- scale(nhanes$LBXNEPCT)
nhanes$eosinophils_z <- scale(nhanes$LBXEOPCT)
nhanes$basophils_z   <- scale(nhanes$LBXBAPCT)
nhanes$lncotinine_z  <- scale(nhanes$ln_lbxcot)         ## to access smoking status, scaled ln cotinine levels


## our y variable - ln transformed and scaled mean telomere length
lnLTL_z <- scale(log(nhanes$TELOMEAN)) 

## our Z matrix
mixture <- with(nhanes, cbind(LBX074LA, LBX099LA, LBX118LA, LBX138LA, LBX153LA, LBX170LA, LBX180LA, LBX187LA, 
                              LBX194LA, LBXHXCLA, LBXPCBLA,
                              LBXD03LA, LBXD05LA, LBXD07LA,
                              LBXF03LA, LBXF04LA, LBXF05LA, LBXF08LA)) 
lnmixture   <- apply(mixture, 2, log)
lnmixture_z <- scale(lnmixture)
colnames(lnmixture_z) <- c(paste0("PCB",c(74, 99, 118, 138, 153, 170, 180, 187, 194, 169, 126)), 
                           paste0("Dioxin",1:3), paste0("Furan",1:4)) 

## our X matrix
covariates <- with(nhanes, cbind(age_z, agez_sq, male, bmicat2, bmicat3, educat1, educat3, educat4, 
                                 otherhispanic, mexamerican, black, wbcc_z, lymphocytes_z, monocytes_z, 
                                 neutrophils_z, eosinophils_z, basophils_z, lncotinine_z)) 

## save
write.csv(lnLTL_z,'y.csv', row.names=FALSE)
write.csv(lnmixture_z,'Z.csv', row.names=FALSE)
write.csv(covariates,'X.csv', row.names=FALSE)

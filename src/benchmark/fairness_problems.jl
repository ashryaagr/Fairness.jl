function communities_crime()
	communities_data = @load_communities_crime
	communities_data[1].Column8 = convert(Array{Float64},
										communities_data[1].Column8) .<= 0.18
	#0.18 is the mean value for racepctblack

	communities_crime_task = Task(communities_data[1], communities_data[2],
								grp=:Column8,
								debiasmeasures=[fpr, fnr])
	FairnessProblem(
		communities_crime_task,
		measures=[fnr, fpr],
		repls=10, nfolds=10, name="communities_crime")
end

function student()
   student_task = Task((@load_student_performance)...,
					   grp=:sex, debiasmeasures=[fnr])
   FairnessProblem(
	   student_task,
	   measures=[fnr, fpr],
	   repls=10, nfolds=10, name="StudentPerformance")
end

function hmda()
	hmda_df = CSV.read(joinpath(@__DIR__, "..", "data", "HMDA_2018_California.csv"))
	filter!(:derived_sex => x -> x != "Sex Not Available", hmda_df)
	categorical!(hmda_df, [:conforming_loan_limit, :derived_loan_product_type ,
			:derived_dwelling_category, :derived_ethnicity, :derived_race,
			:derived_sex, :action_taken, :purchaser_type, :loan_type,
			:applicant_age, ])
	select!(hmda_df, Not(:county_code))
	rename!(hmda_df, "open-end_line_of_credit" => "open_end_line_of_credit")

	hmda_task = Task(deletecols(hmda_df, :action_taken), hmda_df.action_taken,
				grp=:derived_sex, debiasmeasures=[false_omission_rate])

	FairnessProblem(hmda_task,
		measures=[false_omission_rate, fpr],
		repls=10, nfolds=10, name="HMDA_Mortgage")
end

function zafar()
   zafar_task = Task((genZafarData())...,
					   grp=:z, debiasmeasures=[])
   FairnessProblem(
	   zafar_task,
	   measures=[],
	   repls=10, nfolds=10, name="ZafarData")
end

function zafar2()
   zafar2_task = Task((genZafarData2())...,
					   grp=:z, debiasmeasures=[])
   FairnessProblem(
	   zafar2_task,
	   measures=[],
	   repls=10, nfolds=10, name="StudentPerformance")
end

function subgroup()
   subgroup_task = Task((genSubgroupData(setting="B01"))...,
					   grp=:z, debiasmeasures=[])
   FairnessProblem(
	   subgroup_task,
	   measures=[],
	   repls=10, nfolds=10, name="SubGroupData")
end

function biased()
   biased_task = Task((genBiasedSampleData())...,
					   grp=:z, debiasmeasures=[])
   biased = FairnessProblem(
	   biased_task,
	   measures=[],
	   repls=10, nfolds=10, name="BiasedSampleData")
end

function adult()
   adult_task = Task((@load_adult)..., grp=:sex, debiasmeasures=[ppr]) # could keep :sex also as protected attribute
   FairnessProblem(
	   adult_task,
	   refGrp="Male", measures=[ppr, fnr, fpr],
	   repls=10, nfolds=10, name="Adult")
end

function german()
   german_task = Task((@load_german)..., grp=:gender_status, debiasmeasures=[false_omission_rate])
   FairnessProblem(
	   german_task,
	   refGrp="male_single", measures=[foar, fnr, fpr],
	   repls=10, nfolds=10, name="German")
end

function portuguese()
   portuguese_task = Task((@load_bank_marketing)...,
					   grp=:marital, debiasmeasures=[false_omission_rate, ppr])
   FairnessProblem(
	   portuguese_task,
	   measures=[false_omission_rate, ppr, fpr],
	   repls=10, nfolds=10, name="Portuguese")
end

function framingham()
   framingham_df = CSV.read(joinpath(@__DIR__, "..", "data", "framingham.csv"))
   categorical!(framingham_df, [:male, :education, :currentSmoker, :BPMeds,
			   :prevalentStroke, :prevalentHyp, :diabetes, :TenYearCHD])

   framingham_task = Task(deletecols(framingham_df, :TenYearCHD),
			   framingham_df.TenYearCHD, grp=:male,
			   debiasmeasures=[false_omission_rate])

   FairnessProblem(framingham_task,
				   measures=[false_omission_rate],
				   repls=10, nfolds=10, name="Framingham")
end

function loan()
   loan_df = CSV.read(joinpath(@__DIR__, "..", "data", "loan_default.csv"))
   categorical!(loan_df, [:SEX, :EDUCATION, :MARRIAGE,
		   Symbol("default payment next month")])
   loan_task = Task(deletecols(loan_df, Symbol("default payment next month")),
			   loan_df[!, Symbol("default payment next month")],
			   grp=:SEX, debiasmeasures=[false_omission_rate])
   FairnessProblem(loan_task, measures=[false_omission_rate, fpr],
					   repls=10, nfolds=10, name="LoanDefault")
end

function medical()
   medical_df = CSV.read(joinpath(@__DIR__, "..", "data", "meps.csv"))

   categorical!(medical_df, ["REGION","SEX","MARRY","FTSTU","ACTDTY","HONRDC",
			   "RTHLTH","MNHLTH","HIBPDX","CHDDX","ANGIDX","MIDX",
			   "OHRTDX","STRKDX","EMPHDX","CHBRON","CHOLDX","CANCERDX","DIABDX",
			   "JTPAIN","ARTHDX","ARTHTYPE","ASTHDX","ADHDADDX","PREGNT","WLKLIM",
			   "ACTLIM","SOCLIM","COGLIM","DFHEAR42","DFSEE42", "ADSMOK42",
			   "PHQ242","EMPST","POVCAT","INSCOV", "UTILIZATION"])

   medical_task = Task(deletecols(medical_df, Symbol("UTILIZATION")),
			   medical_df[!, Symbol("UTILIZATION")],
			   grp=:RACE, debiasmeasures=[false_omission_rate])
   FairnessProblem(medical_task, measures=[false_omission_rate, fnr],
					   repls=10, nfolds=10, name="MedicalExpenditure")
end

fairness_problems() = [compas(), adult(), german(), portuguese(),
					   student(), framingham(), loan(), medical()]

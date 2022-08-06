# LOAN-ELIGIBILITY-PREDICTION
Loans are fundamental to the banks' operations. The principal profit is directly derived from the interest of the loan. After an exhaustive verification and validation procedure the loan firms provide a loan. However, if the borrower can return the loan without any issues, it still has no guarantee. Creating a predictive model in this project to detect if the applicant can reimburse the lenders or not. We prepare the data using Jupyter notebooks to forecast the target variable using several models. This project has been implemented using Machine learning techniques using Linear Regression with an accuracy of 80 percent. 

 AIM AND SCOPE OF THE PROJECT:
In our country there is a shortage of surveillance systems due to high cost. To overcome this shortage we came up with this technique.The main goal is to predict if the person can repay the loan or not.

MODULES OVERVIEW:

Module is a logical separation of functionality within a project. They are basically used for reusability and better code maintenance. There are nine modules used here,
Data Cleaning
Feature engineering
Model Training

SCREENSHOTS:

![download](https://user-images.githubusercontent.com/65251355/183254285-f3f45665-98a5-4347-a812-b3bb0fd34801.png)





![download (1)](https://user-images.githubusercontent.com/65251355/183254631-925527c1-8993-4897-acc3-503afd4a2997.png)
![download (2)](https://user-images.githubusercontent.com/65251355/183254691-cfef50a9-ea57-476c-a10c-911c150a8ff8.png)
![download (3)](https://user-images.githubusercontent.com/65251355/183254699-ab8695e6-c07f-4c4e-950b-6dbdaf287422.png)
![download (4)](https://user-images.githubusercontent.com/65251355/183254702-9a60d659-0aa1-4d11-8644-6a0decdc5b1c.png)

From the above plots we can come to know that :-
80% of loan applicants are male.
~65% of applicants are married.
15% of applicants are self-employed.
85% of applicants have repaid their pervious loans.

![download (5)](https://user-images.githubusercontent.com/65251355/183254730-545d95f7-6712-489e-ba80-5c7f807b5236.png)
![download (6)](https://user-images.githubusercontent.com/65251355/183254734-7e2dc6e7-6a7f-4514-ae72-9ea253d60316.png)
![download (7)](https://user-images.githubusercontent.com/65251355/183254739-00002a6a-e1e6-4d4a-b07b-89d2053e6c61.png)

The following observations are made :-
Most applicants have no dependants
~80% of applicants are graduates
Most applicants have property in semi-urban areas

![download (8)](https://user-images.githubusercontent.com/65251355/183254771-660f1406-fd5c-42c5-be9f-4b79c4064ddc.png)

It can be inferred from the above plot that the income of the applicants are not equally distributed. We will try to normalize it later as algorithms tend to work better on equally distributed data.

![download (9)](https://user-images.githubusercontent.com/65251355/183254797-78a58930-1739-47c4-8fbf-af15331a0139.png)

This plot tells us about the varying levels of outliers or extreme values which are present in the data. It also explains the disparity present in the income of the applicants. This maybe due to the reason we are looking at people from different educational backgrounds. We will now segregate them based on their educational qualifications.

![download (10)](https://user-images.githubusercontent.com/65251355/183254819-76055a19-5895-484c-8ba0-2ee85efa2624.png)

We can see that the higher number of graduates were the reason for the outliers. We also can note that graduated people tend to have higher income than the other category.

![download (11)](https://user-images.githubusercontent.com/65251355/183254859-a254cd58-4b30-462e-8ae8-9ec6512b4b7f.png)
![download (12)](https://user-images.githubusercontent.com/65251355/183254873-69c9d704-b820-4436-963b-58781e1c659a.png)

We see a distribution which is similar to the previous one. We also see a high amount of outliers.

![download (13)](https://user-images.githubusercontent.com/65251355/183254906-93b011a3-b738-4369-8e69-a37c52504348.png)

It is clear that the portion of female and male applicants whose loans got approved are more or less the same.
![download (14)](https://user-images.githubusercontent.com/65251355/183254919-988873ad-fbd2-4d0d-9283-7ad108a84769.png)
![download (15)](https://user-images.githubusercontent.com/65251355/183254931-e48b3b5e-ad95-4e36-b793-f4f74d64fbfc.png)
![download (16)](https://user-images.githubusercontent.com/65251355/183254937-ef1a01f1-72f7-42c4-b415-09a6b0f24a26.png)
![download (17)](https://user-images.githubusercontent.com/65251355/183254944-352875c5-f56b-4b43-995d-3f61bc2e3953.png)

From the above graphs we can understand the following :-
The number of approved loans is higher for married applicants.
Applicants with 1 or 3+ dependants have the same chance to get their loans approved.

![download (18)](https://user-images.githubusercontent.com/65251355/183254972-91604524-e4ee-45aa-a108-6cb18748bf0c.png)
![download (19)](https://user-images.githubusercontent.com/65251355/183254976-185d9e87-898b-4cbb-9be4-552ed305eb2f.png)

By plotting the remaining categorical variables against Loan_Status :-
There is a huge chance for the loan to be approved if the applicant doesn't have any credit history.
Interestingly, the chancee of getting loan approved if the property is present in semi-urban area is more than rural and urban areas.

![download (20)](https://user-images.githubusercontent.com/65251355/183255030-9b43dc4d-59bf-4839-a806-41e53d5608bb.png)

It is clear that the ApplicantIncome doesn't have any influence over Loan_Status.
We will do a similar analysis on the CoapplicantIncome.
![download (21)](https://user-images.githubusercontent.com/65251355/183255038-be5990ce-82f3-4721-8190-d2a85d67c381.png)

This shows that, if the CoapplicantIncome is low then there is a higher chance for the loan to be approved, but that doesn't make any sense. The reason for this maybe due to many applicant's being not married. Hence, the CoapplicantIncome in their data will be 0.
We will now create a new numerical variable, TotalIncome and see whether it has some relationship with Loan_Status.

![download (22)](https://user-images.githubusercontent.com/65251355/183255073-da35be0c-f149-469c-9cd4-619d9a5fda28.png)

Now we can see that, the chance of the loan to be approved is low if the Total_Income is low, which makes sense.
Now let's visualize the relationship between LoanAmount and Loan_Status

![download (23)](https://user-images.githubusercontent.com/65251355/183255091-64d41c36-76fd-485d-b318-eb53c80e51f2.png)

Clearly, there is a higher chance for the loan to be approved, if the LoanAmount is lower.
Now let's drop the columns which we previously created for data exploration. When we are at it we well change 3+ data values in Dependants column to 3, so that it becomes a numerical variable. We will also change the target variable to 1 and 0, since many logisitc regression algorithms depend on a numerical target variable.

![download (24)](https://user-images.githubusercontent.com/65251355/183255137-e902dca6-0913-4175-891b-5b3f94e728a1.png)
![download (25)](https://user-images.githubusercontent.com/65251355/183255144-93c14075-02aa-4964-a4f0-b4cc814b4e9d.png)

The distribution is much more tighter and most of the outliers which were previously there has been removed.
 
The dataset is now ready to model training and testing.
 

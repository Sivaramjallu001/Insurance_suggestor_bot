import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from collections import deque
import torch


# ✅ Gemini API setup
genai.configure(api_key="YOUR-API-KEY")  # Replace with your key
model = genai.GenerativeModel("gemini-1.5-flash")


# ✅ Step 1: Store policies
policy_1_text = """Shop Keeper

Shop Keeper

This is a package policy specially designed for small shopkeepers. It is a single policy combining the various insurance requirements of shopkeepers.
Highlights
This is a package policy specially designed for small shopkeepers. It is a single policy combining the various insurance requirements of shopkeepers.
Discount in premium is available if a minimum number of four sections is selected including Section I (b).
Only one policy can be taken by one shopkeeper for each shop in a specific location having separate books of accounts.
Scope
The policy comprises of following 11 sections :
Section I - Building & Contents
1. Covers shop building and/or contents therein against loss or damage caused by Fire & Allied perils i.e.
2. Fire, lightning, explosion of gas in domestic appliances
3. Bursting and overflowing of water tanks, apparatus or pipes
4. Aircraft or articles dropped therefrom
5. Riot, strike, malicious damage, terrorist act
6. Earthquake-Fire and/or shock, subsidence and landslide (including rockslide)
7. Flood, Inundation, Storm, Tempest, Typhoon, Hurricane, Tornado or Cyclone.
8. Impact damage by rail/road vehicle not belonging to the insured.
Section Il - Burglary & Housebreaking
Covers contents of insured shop premises(excluding money and valuables) against loss or damage by burglary and/or housebreaking.
Section Ill - Money Insurance
Covers loss of money in transit, loss of money/valuables whilst contained in a locked safe, loss of money contained in cashier's till and/or counter by burglary/housebreaking.
Section IV - Pedal - Cycles
1.  Covers loss/damage to pedal cycles belonging to insured by:
2.  Fire, lightning or external explosion.
3.  Riot, strike, malicious or terrorist act.
4.  Burglary and/or Housebreaking or theft
5.  Accidental external means
6.  Flood, cyclone, storm, tempest, and other similar convulsions of nature and atmospheric disturbance
7.  Earthquake Fire and shock
This section also covers legal liability of insured for death/injury to third parties or damage to their property arising out of use of the insured pedal cycles.
Section V - Plate Glass
Covers loss of or damage to fixed plate glass in insured's shop by accidental means.
Section VI - Neon Sign/Glow Sign
1. Covers loss of or damage to neon sign/ glow sign by
2. Accidental external means
3. Fire, lightning or external explosion or theft.
4. Riot, strike, malicious or terrorist act
5. Flood, inundation, storm, tempest, typhoon, hurricane, tornado, cyclone.
Section Vll - Baggage
Covers loss or damage to accompanied personal baggage of insured or baggage in connection with his trade, whilst anywhere in India, by accident or misfortune.
Section Vlll - Personal Accident
Covers insured and spouse and/or his children, named in the schedule and aged between 5&70 years, against bodily injury caused solely and directly by accident and resulting in death or permanent total or partial disablement or temporary total disablement within 12 calendar months of such injury.
Section IX - Fidelity Guarantee
Covers direct pecuniary loss suffered by the insured due to fraud or dishonesty committed by any of insured's salaried employees.
Section X - Public Liability
Covers
1. Legal liability in respect of accidental death or bodily injury to a third party or accidental damage to their property during performance of any act in connection with insured's business.
2. Compensation to insured's employees under Workmen's Compensation Act or Common Law.
Section Xl - Loss of Profit
Covers loss of profit due to interruption of business consequent upon loss or damage sustained by property insured under Section I of the policy due to insured perils.
It is necessary to opt for a minimum of 4 sections for this policy to be issued of which Sections l&ll are compulsory.
Who can take the policy?
This policy can be taken by small shopkeepers whose shop building value and contents value does not exceed INR 10 lacs. In case it exceeds INR 10 lacs, this policy cannot be given.
This policy is meant for shops only. Mere registration under Shops and Establishment Act does not entitle the premises to be insured under this policy. Hence Restaurants and Tea /Coffee shops cannot be insured under this policy. However,shops selling goods where minor repair work is carried on incidental to the main business of selling, can take this policy.
How to select the sum insured?
The shop building should be insured on market value basis i.e.depreciated value basis.The contents should be insured on cost price basis. The sum insured for contents under Section l&ll should be identical. The sum insured under Sections is limited to specified percentage of the sum insured for contents.
How to claim?
In case of any incident giving rise to a claim under this policy, please take the following steps:
1.  Take necessary steps to minimize the loss/damage.
2.  In case of fire, inform fire brigade immediately.
3.  In case of theft, larceny or burglary inform the police immediately along with a list of items stolen and their approximate value.
4.  Inform insurance company by phone or fax and in writing.
5.  Extend full co-operation to the surveyor appointed by the insurance Co. and provide necessary documents to substantiate the loss. A claim form issued by the company is also to be submitted.
6.  In case any rights of recovery exist against any other party responsible for the loss, your rights of recovery have to be subrogated to the insurance company on payment of claim.
Note:Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

How to claim?
In case of any incident giving rise to a claim under this policy, please take the following steps:
1. Take necessary steps to minimize the loss/damage.
2. In case of fire, inform fire brigade immediately.
3. In case of theft, larceny or burglary inform the police immediately along with a list of items stolen and their approximate value.
4. Inform insurance company by phone or fax and in writing.
5. Extend full co-operation to the surveyor appointed by the insurance Co. and provide necessary documents to substantiate the loss. A claim form issued by the company is also to be submitted.
6. In case any rights of recovery exist against any other party responsible for the loss, your rights of recovery have to be subrogated to the insurance company on payment of claim.
Note:Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.
Product FAQ
	
	 
	
	 
	FAQ
	 
	1. What is Shopkeeper Insurance for?
This is a package policy specially designed for small shopkeepers. It is a single policy combining the various insurance requirements of shopkeepers.
2. Who can take the Shopkeeper Insurance?
This policy can be taken by small shopkeepers whose shop building value and contents value does not exceed INR 10 lacs. In case it exceeds INR 10 lacs, this policy cannot be given.
3. What all risks are covered under the policy?
The policy consists of 11 Sections:
 Section I - Building & Contents
 Section Il Burglary & Housebreaking
 Section Ill - Money Insurance
 Section IV - Pedal - Cycles o Section V - Plate Glass
 Section VI - Neon Sign/Glow Sign
 Section Vll - Baggage o Section Vlll - Personal Accident
 Section IX - Fidelity Guarantee
 Section X - Public Liability o Section Xl - Loss of Profit
4. What should be included under the policy cover?
The shop building should be insured on market value basis i.e. depreciated value basis. The contents should be insured on cost price basis. The sum insured for contents under Section I & Il should be identical. The sum insured under Sections Ill, V, VI, Vll, X & Xl is limited to specified percentage of the sum insured for contents.
5. Can I opt out of any sections covered under the policy?
It is necessary to opt for a minimum of 4 sections for this policy to be issued of which Sections I & Il are compulsory.
6. Can I opt for 'Loss of profit' (Sec-12) section without taking Sec 1 & 2?
 
No, insured has to take Section I & Il for the LOP section coverage."""
policy_2_text = """Rasta Apatti Kavach Policy
 

Rasta Apatti

Kavach Policy



Highlights

The policy offers PERSONAL ACCIDENT compensation cover including reimbursement of Hospitalization expenses incurred due to an accident.

Scope of Cover

1. Section I - The policy offers Personal Accident compensationcover for Sum Insured ranges from INR 25000 to INR 1 lac and in further multiples of INR 1 lac upto INR 10 lac.

2. Section Il Hospitalization Expenses for bodily injury caused by and arising out of an accident

a. Road Accident (at additional premium)

b. Arising out of and during the course of employment (if opted for at additional premium )

c.  Any other accident (wider cover) (if opted for at an additional premium)

There is also anoption to cover at an additional premium, the Hospitalization Expenses for bodily injury caused by and arising out an accident to Third Parties arising out of a motor accident.

Sum Insured

The Sum Insured ranges from INR 25000 to INR 1 lac and in further multiples of INR 1 lac upto INR 10 lacs for Personal Accident as well as Hospitalization expenses. The hospitalization limit allowed shall be equal to and upto the maximum Accident limit.

Sum Insured

The Sum Insured ranges from INR 25000 to INR 1 lac and in further multiples of INR 1 lac upto INR 10 lacs for Personal Accident as well as Hospitalization expenses. The hospitalization limit allowed shall be equal to and upto the maximum Accident limit.

Premium

Premium depends upon the Sum Insured chosen.

Policy is also available on group basis.

The above is only broad indication of a cover offered. For further details contact any of our Policy issuing office.

Note:Policy details given are indicative, not exhaustive. Please contact your nearestNlA office for further details.



1.  Immediate intimation of accident to insurer.

2.  Intimation to Police in the event of rail/road accident.

3.  Submission of claim form giving details of occurrence and description of disability and supporting documents.

Product FAQ



FAQ

 



1. Is the cover for only Road Accidents ?

No. there is cover for employment related accidents and other accidents,

2. Does the policy cover medical expenses ?

Yes, the policy covers hospitalization expenses arising out of accidents.

3. Is this policy annual ?

The policy can be taken annually and is also available on long term basis for 3 years maximum.

4. Can I cover my staff members / Employees also ?

http://bookstack.zubera.one/link/330#bkmrk-yes%2C-a-group-policy-
 
Yes, a group policy can be taken to cover employees / staff members."""
policy_3_text = """Plate Glass Insurance
Plate Glass Insurance

Product Highlights





Highlights

It is an annual policy that covers all kinds of accidental breakages of the plate glass fixed to display windows or show cases of commercial establishments.

Scope

The policy covers the cost of making good accidental breakage of insured glass by any reason whatever, except those that are specifically excluded.

Exclusions

1.  Fire or explosion

2.  Earthquakes or such other convulsion of nature

3.  Damage to frames

4.  Cost of removal or replacement of any fittings or fixtures necessitated for replacing the broken glass

5.  Cracked or imperfect glass

6.  Any superficial damage or scratching

Who can take the policy?

Any person who installs plate glass of substantial value can avail of this policy.

Premium

Rate of premium depends on the type of glass, situation, previous experience and neighbourhood.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.



Claims Process:

1. Immediate notification of the loss to be given to Insurer.

2. Submission of claim form giving description of loss and supporting documents.

3. Furnish all such information and documentary evidences as the Company may require.

4. Insured shall take all reasonable precautions to protect the glass insured hereunder and in the event of it being exposed to unusual risk on account of any procession, show, building alteration or repairs or other special circumstances the insured shall cause it to be adequately protected by boards or otherwise.

Product FAQ



FAQ

 



FAQs:

1.  Who can take this policy?

Anyone can take the policy for the plate glass having substantial value.

2.  What is covered under the policy?

The policy covers loss or damage to the plate glass occasioned by breakage-related risks only.

3.  What is meant by breakage?

The word "breakage" will not include scratches, disfiguration, dis-coloration, or damage other than a fracture extending through the entire thickness of the glass.

4.  Are the window frames & fittings covered?

No, window frames & fittings are not covered.

5.  What is the maximum indemnity in the policy?

The maximum indemnity is the value of the glass at the time of occurrence of loss or the insured's estimate of value mentioned in the policy, whichever is less.

6.  What is not covered under the policy?

The following common losses are not covered under the policy:

 Breakage caused directly or indirectly through fire, gas, heat, or any loss that is covered by a fire policy.

 Earthquake, volcanic eruption, cyclone, or such other convulsions of nature, war and allied perils, riot, and strike.

 Loss/damage to window frames and fittings.

 Cracked or imperfect glass or scratches on the glass.

 Willfully caused by the insured.

 Embossed, silvered, lettered, bent, or any special type of glass other than plain or of ordinary glazing quality unless declared and expressly insured by the policy.  Consequential loss arising out of breakage.

o Nuclear exclusion."""
policy_4_text = """Jewellers Block Insurance


Scope

The policy comprises four sections which are optional except for section I which is compulsory.

Section I : Covers loss or damage to jewellery , gold and silver ornaments or plates , pearls, precious stones, cash and currency notes whilst contained in the premises insured, by fire,explosion, lightning,burglary,house breaking, theft, hold up, robbery, riot, strike and malicious damage and terrorism.

Section Il : Covers loss or damage to jewellery, gold etc. as described in Section I whilst it is in the custody of the insured, his/her partners, employees, directors, sorters of diamonds or whilst such property (excluding cash and currency notes) is in the custody of brokers, agents, cutters and goldsmiths.

Section Ill : Covers loss or damage to property described in Section I whilst in transit by registered parcel post, air freight or through angadia.

Section IV : Covers loss or damage to trade and office furniture and fixtures in insured premises due to fire,explosion, lightning,burglary,house breaking, theft, hold up, robbery, riot, strike and malicious damage and terrorism.

Highlights

This is a package policy specially designed for jewellers & diamontaires i.e. those establishments dealing solely in diamonds.

Jewellers premises are categorised into Class l, Il or Ill depending upon the type of security provided for the premises.

Jewellers



Block Insurance

Discount in premium is available in case the premises have special protection devices like built-in vaults, strong rooms, closed circuit T.V. or armed guards.

Who can take the policy?

The policy can be taken by jewellers who are wholesalers or retailers. The policy cannot be given to establishments whose work is predominantly manufacturing like cutters and goldsmiths. The policy also cannot be given to angadias , brokers or pawn brokers etc.

How to select the sum insured?

The sum insured under Section I and Il should represent the cost price of the jewellery items. The sum insured under Section Ill should represent the maximum loss likely, arising out of any one incident. The sum insured under Section IV should represent the market value of the property.

How to claim?

In case of any incident giving rise to a claim under the policy , the following steps should be taken

1. Inform insurance company within 24 hrs.

2. In case of burglary,theft etc. inform police immediately and obtain FIR

3. Submit claim form and relevant documents to surveyor appointed by Insurance Co. to su bstantiate loss test.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

1.  Immediate notification of the loss to be given to Insurer.

2.  Immediate intimation to the Police and filing a FIR event of burglary / theft .

3.  Submission of claim form giving description of loss and supporting documents.



1. Who can take this cover ?

This cover is designed to be taken by Jewellers for covering their stocks and property in premises.

2. Can other business dealing in Jewellery take this cover ?

Dimontaries and jewellery manufacturers can also take this cover.

3. How to decide the sum insured to be taken ?

http://bookstack.zubera.one/link/344#bkmrk--2"""
policy_5_text = """House Holder Insurance


Scope

The policy comprises of 10 sections as given here under

Section I - Fire & Allied Perils

1. Coverage for building

2. Covers contents of the dwelling belonging to the proposer and his/her family members permanently residing with him/her.

Allied Perils:

1.  Fire, Lightening, Explosion of gas in domestic appliances

2.  Bursting and overflowing of water tanks, apparatus or pipes.

3.  Damage caused by Aircraft

4.  Riot, Striker Malicious or Terrorist Act

5.  Earthquake, Fire and/or Shock, subsidence and Landslide (including Rockslide) damage

6.  Flood, Inundation, Storm, Tempest, Typhoon, Hurricane, Tornado or Cyclone.

7.  Impact damage

Section Il - Burglary & House Breaking including larceny and theft.

Covers contents of the dwelling against loss due to burglary, house breaking, larceny or theft.

Section Ill - All Risks (Jewellery & Valuables)

Covers loss or damage to your jewellery and valuables by accident or misfortune whilst kept, worn or carried anywhere in India subject to the value declared in the schedule.

Section V - Breakdown of Domestic appliances

Covers domestic appliances against unforeseen and sudden physical damage due to mechanical or electrical breakdown.

Section VI - T.V. Set including VCP/VCR (ALL RISKS)

Covers loss or damage to T.V.Set including VCP/VCR by fire and allied perils, burglary, house breaking or theftr breakage due to accidental external means, mechanical or electrical breakdown. Any legal liability arising out of bodily injury or accidental death of any person other than insured's family members or employee as also damage to property not belonging to or in the custody of insured , caused by use of the T.V. Set is also covered upto a limit of INR 25,000/ .

Section Vll - Pedal Cycles (All Risks)

Covers loss or damage to pedal cycles by :

1.  Fire & allied perils

2.  Burglary, housebreaking, theft

3.  Accidental external means

4.  Third party personal injury or Third party property damage for INR 10,000/

Section VII - Baggage Insurance

Covers loss or damage to insured's accompanied baggage by accident or misfortune whilst the insured is traveling on tour or holiday anymv'here in India.

Section IX - Personal Accident

Covers Death or bodily injury by accidental, violent, external and visible means to the insured person named in the schedule and subject to limits specified therein.

Section X - Public Liability

Covers Insured's legal liability for bodily injury or loss of or damage to property of third party limited to amount specified in the schedule and workmen's compensation liability to domestic servants engaged in insured's premises.

It is compulsory to opt for Section 1B of the policy. A minimum of three sections including Section 1B have to be taken for issuance of this policy.

This is a package policy specially designed to meet the insurance requirements of a householder.

Highlights

This is a package policy specially designed to meet the insurance requirements of a householder by combining under a single policy, a number of our standard policies usually taken by householders.

Discount in premium is offered depending upon the number of sections of the policy, opted for, by the proposer.

How to select the sum insured?

For the insurance of household items, it would be necessary to group the items in a broad category like furniture, clothing , linen, utensils , crockery etc. and give a value equivalent to the market value i.e. the value for which this used item could be bought or sold in the market.

Sections I A & B, Il, Ill, IV, VI ,VII & Vlll should be insured on market value basis as described above.

It is a condition of Section V i.e. breakdown of domestic appliances , that the sum insured should represent the current replacement value of a similar item. For e.g. to insure 165 ltr. Godrej fridge which is 3 years old, the sum insured should be equivalent to the cost price of a new 165 ltr. Godrej fridge.

However, the claim amount payable would be the amount required to bring the damaged item to the same condition as it was prior to the damage subject to the adequacy of the sum insured.

The sum insured under section IX i.e. Personal Accident should not exceed 72 months salary from gainful employment.



How to Claim?

In case of any incident leading to a valid claim under the policy, following steps should be taken:

1.  Take necessary steps to minimize the loss/damage.

2.  In case of fire, inform fire brigade immediately.

3.  In case of theft, larceny or burglary inform the police immediately along with a list of items stolen and their approximate value.

4.  Inform insurance company by phone or fax and in writing.

5.  Extend full co-operation to the surveyor appointed by the insurance Co. and provide necessary documents to the substantiate the loss. A claim form issued by the company is also to be submitted.

6.  In case any rights of recovery exist against any other party responsible for the loss, your rights of recovery have to be subrogated to the insurance company on payment of claim,

Note:Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

1.  Immediate notification of the loss to be given to Insurer.

2.  Immediate intimation to the Police and filing a FIR ,in event of burglary / theft .

3.  Submission of claim form giving description of loss and supporting documents.



1 . What is Householder Insurance for ?

It covers Building and Contents of a house against various risks / perils.

2. Can I take this policy if I am staying on Rent ?

In that case one can take cover for Contents only.

3. What should be included under the policy cover ?

All the contents including Furniture, Fixtures, Fittings, Domestic Appliances, Jewellery etc. can be covered.

4. Does the policy cover pets in the house ?

No. Only properties in the house are covered.

5. What all risks are covered under the policy ?

http://bookstack.zubera.one/link/346#bkmrk-policy-covers-variou
 
Policy covers various risks including Fire, Natural perils, Theft & Burglary, Breakdown of appliances, Pedal Cycles, Personal accident etc."""
policy_6_text = """PROSPECTUS New India Pension Protect Personal Accident Policy
NEW INDIA PENSION PROTECT PERSONAL ACCIDENT POLICY

(UIN No: NIAPAlP23064V012223)

1. PREAMBLE
This Policy is a contract of insurance issued by The New India Assurance (hereinafter called the 'Company') to the proposer mentioned in the schedule (hereinafter called the 'Insured') to cover the person(s) named in the schedule (hereinafter called the 'Insured Persons The policy is based on the statements and declaration provided in the proposal Form by the proposer and is subject to receipt of the requisite premium.

2. OPERATIVE CLAUSE
Any amount payable under the policy shall be subject to the terms of coverage, exclusions, conditions and definitions contained herein. Maximum liability of the Company under all such Claims during each Policy Year shall be the Sum Insured specified in the Schedule.

3.       POLICY COVERAGE Basic Cover:

Accidental Death-Full Sum Insured

Built In Covers:

Carriage of Dead Body— I % of CSI maximum Rs. 2500/Funeral Expenses — Rs. 2500/-

4.       DEFINITIONS:

4.1  ACCIDENT

An accident is a sudden, unforeseen, involuntary event caused by external, visible and violent means.

4.2  INJURY

Injury means accidental physical bodily harm excluding illness or disease solely and directly caused by external, violent and visible and evident means which is verified and certified by a medical practitioner.

4.3  SUM INSURED

Maximum sum insured allowable is 72 times of Monthly Pension Lost (as per the Pension Scheme of the respective employer / Annuity Plan purchased by the Employer / Annuity Plan purchased by the retiree ) in the event of Death of the Pensioner.

Pension Lost implies the difference between the derived pension as on the date of commencement of the policy period and the family pension available to such dependent family members as would become recipient of the family pension as recorded in the Pension Scheme / Annuity Plan after the death of the pensioner.

4.4  FAMILY PENSION

Family Pension is defined as a regular monthly amount which an employer / insurer of the Annuity Plan would pay to a person who belongs to the family of the employee in the event of the Death of the employee.

4.5  AGE

Entry Age - From the time of becoming a Pensioner, but not less than 35 years and up to age of 70 years. Age means age ofthe Insured Person on the last birthday as on the date ofcommencement of the policy.

(Note: On completion of 65 years of age the acceptance of new proposals would be subject to submission of satisfactory physical fitness certificate from family doctor / medical practitioner and premium loading @ 2 % every year.)

4.6  POLICY PERIOD means period of one year for which the policy is issued.

4.7  GRACE PERIOD

Grace period (30 days or as amended by IRDA from time to time) means the specified period of time immediately following the premium due date during which a payment can be made to renew or continue a policy in force without loss of continuity benefits. Coverage is not available for the period for which no premium is received by the Company.

4.8  RENEWAL

Renewal defines the terms on which the contract of insurance can be renewed on mutual consent with the provision of grace period for treating the renewal continuous.

4.9  BENEFIT PAYABLE -

If at anytime during the currency of this policy, the Insured person shall die

(a) resulting solely and directly from Accident , then the Company shall pay to such dependent family members as would become recipient of the family pension as recorded in the Pension Scheme / Annuity Plan purchased by the Retiree which is the basis of this policy contract. (b) Provided such death shall have occurred within Twelve months of the date of such Accident.

Basic Cover:

Accidental Death-Full Sum Insured

Built In Covers
Carriage of Dead Body— 1 % of CSI maximum Rs. 2500/Funeral Expenses — Rs. 2500

5. EXCLUSIONS

The Company shall not be liable to make any payment under this policy in respect of any Benefit for Death of the Insured from

(a)    From intentional self —injury

(b)    From suicide

(c)    From voluntary self exposure to sports / hazardous activities / adventure sports /Adventure activities

(d)    Any claim arising due to illness

(e)    Whilst under the influence of intoxicating liquor or drugs

(f)     Whilst engaging in Aviation or Ballooning ,whilst mounting into or dismounting from or travelling in any balloon or aircraft other than as a passenger (fare paying or otherwise) in any licensed standard type of aircraft anywhere in the world

'Standard Type of Aircraft' means any aircraft duly licensed by appropriate authority to carry passengers (for hire or otherwise) irrespective of whether such an aircraft is privately owned OR chartered OR operated by a regular airline OR whether such an aircraft has a single engine or multi-engine.

(g)    Directly or indirectly caused by any disease, veneral disease / diseases or insanity"

(h)    Circumcision or Strictures or Vaccination or Inoculation or change of life or beauty treatment of any description or dental or eye treatment or dissipation or nervous breakdown (which expression shall also cover general debility (rundown conditions and general overhaul ) or veneral disease or intemperance .

(i)     Arising or resulting from the Insured committing any breach of law with criminal intent.

(j)     Arising out of directly or indirectly connected with or traceable to —War,Invasion,Act of foreign enemy, Hostilities (whether war be declared or not ) , Civil war, Rebellion, Insurrection, Mutiny,Military or Usurped power Seizure, Capture, Arrests, Restraints and Detainment by any kings, princes and people of whatever nation, condition or quality.

(k)    Any claim resulting or arising from or any consequential loss directly or indirectly caused by or contributed to or arising from:

A.   Ionizing radiation or contamination by radioactivity from any nuclear fuel or from any nuclear waste from the combustion of nuclear fuel or from any nuclear waste from combustion (including any self-sustaining process of nuclear fission) of nuclear fuel.

B.   Nuclear weapons material

C.   The radioactive, toxic, explosive or other hazardous properties of any explosive nuclear assembly or nuclear component thereof.

D.   Nuclear, chemical and biological terrorism

(l) Any loss arising out of the Insured Person's actual or attempted commission of or wilful participation in an illegal act or any violation or attempted violation of the law.

6.      CLAIMS PROCEDURE & CONDITIONS 

6.1.   Notification of claim:

i.        Intimation about an event or occurrence that may give rise to a claim under this policy must be given within 30 days of its happening.

ii.      Claims for insurance benefits must be submitted to the Company not later than one (1) month after transportation of the mortal remains/ burial in the event of Death.

Note:

1.    The Company will examine and relax the time limit mentioned herein above depending upon the merits of the case.

2.    Proof satisfactory to the Company shall be furnished of all matters upon which a claim is based. Any medical or other agent of the Company shall be allowed to examine or have post-mortem examination of the Insured, as may reasonably be required on behalf of the Company. Such evidence as the Company may from time to time require shall be furnished and the post-mortem examination report if necessary, be furnished within the space of fourteen days after the demand is raised in writing.

3.    In case of death of the insured person the policy automatically ceases to be operative, without any refund of premium under any circumstances.

4.    No sum payable shall ordinarily carry any interest. In case of any extra ordinary delay on the part of insurer ,such claims shall be paid by the insurer as specified in IRDA (Protection of Policyholder's Interest) regulations 2017 dated 22.06.2017

5.    The Company shall not be liable to make any payment under this policy in respect of any claim if such claim be in any manner fraudulent or supported or by any fraudulent statement or device, whether by the Insured or by any person on behalf of the Insured.

6.2.   Documents required for processing a claimBasic documents required for claims

i. Duly completed claim form ii. Photo Identity Proof of the insured person ili. Copy of FIR/ Panchnama / Police Inquest Report (wherever these reports are required as per the circumstance of the Accident) duly attested by the concerned Police Station iv. Copy of Medico Legal Certificate (wherever it is required as per the circumstance of the Accident) duly attested by the concerned Hospital

v. Death certificate; vi. Post Mortem Report (if conducted); vii. Identity proof of Nominee or Family Pension Recipient Original Succession

Certificate / Original Legal Heir Certificate or any other proof to the satisfaction of the Company for the purpose of a valid discharge in case nomination is not filed by deceased. Any other relevant document required by the Company for assessment of the claim viii. Any other relevant document required by the Company for assessment of the claim

6.3.   Payment of claim

All claims under the policy shall be payable in Indian currency only.

6.4.   Claim Settlement

i.               The Company shall settle or reject a claim, as the case may be, within 30 days from the date of receipt of last necessary document.

ii.              In case of delay in the payment of a claim, the Company shall be liable to pay interest to the policyholder from the date of receipt of last necessary document to the date of payment of claim at a rate 2% above the bank rate.

iii.            However, where the circumstances of a claim warrant an investigation in the opinion of the Company, it shall initiate and complete such investigation at the earliest, in any case not later than 30 days from the date of receipt of last necessary document. In such cases, the Company shall settle or reject the claim within 45 days from the date of receipt of last necessary document.

iv.             In case of delay beyond stipulated 45 days, the Company shall be liable to pay interest to the policyholder at a rate 2% above the bank rate from the date of receipt of last necessary document to the date of payment of claim.

(Explanation: "Bank rate" shall mean the rate fixed by the Reserve Bank of India (RBI) at the beginning ofthe Financial Year in which claim has fallen due)

7.      RENEWAL

The policy shall ordinarily be renewable except on grounds of fraud, misrepresentation by the insured person.

The Company shall endeavour to give notice for renewal. However, the Company is not under obligation to give any notice for renewal.

Request for renewal along with requisite premium shall be received by the Company before the end of the policy period.

At the end of the policy period, the policy shall terminate and can be renewed within the Grace period of 30 days to maintain continuity of benefits without break in policy. Coverage is not available during the grace period.

The cover for the Insured shall terminate immediately in the event of admissible claim and settlement of 100% Sum Insured under Accidental Death Coverage and no Renewal of contract will be permissible.

The Insured shall give immediate notice to the Company of any change in status of the pensioner or source of income of the insured person, other than the pension.

The Insured shall, on tendering any premium for the renewal of this policy, give notice in writing to the of any disease, physical defect or infirmity with which he has become affected since the payment of last preceding premium.

This policy may be renewed by mutual consent every year and in such event the renewal premium shall be paid to the Company on or before the date of expiry of the policy or of the subsequent renewal thereof. The Company shall not ,however, be bound to give notice that such renewal premium is due.

Possibility of revision of the premium rates:

The company, with prior approval of IRDA], may revise or modify the premium rates.

8. CANCELLATION

The Insured may cancel this Policy by giving 15days' written notice, and in such an event, the Company shall refund premium on short term rates for the unexpired Policy Period as per the rates detailed below.

The premium, on cancellation by insured, will be retained BY INSURER as follows:

Period On Risk

Rate Of Premium To Be Retained



Up to one month

1/4th of the annual rate

Up to three months

1/2 of the annual rate

Up to six months

3/4th of the annual rate

Exceeding six months

Full annual rate

i) Notwithstanding anything contained herein or otherwise, no refunds of premium shall be made in respect of Cancellation where, any claim has been admitted or has been lodged or any benefit has been availed by the Insured person under the Policy.

ii. The Company may cancel the Policy at any time on grounds of misrepresentation, non- disclosure of material facts, fraud by the Insured Person, by giving 15 days' written notice. There would be no refund of premium on cancellation on grounds of misrepresentation, non-disclosure of material facts or fraud.

9. NOMINATION
The insured person is required at the inception of the policy, to make a nomination of the eligible family pension recipient / recipients for the purpose of payment of claim under the policy in the event of Death of the policyholder. Any change in the nomination shall be communicated to the insurer in writing and such change shall be effective only when endorsement on the policy is made. In the event of death of the policyholder, Company will pay the nominee ( as named in the policy / pension scheme ) and in case there is no nominee ,to the legal representative of the policyholder whose discharge shall be treated as full and final discharge of its liability under the policy.

10. POLICY DISPUTES
Any dispute concerning the interpretation of the terms, conditions, limitations and/or exclusions contained herein is understood and agreed to by both the Insured and the Company to be subject to Indian Law.

11. ARBITRATION
If any dispute or difference shall arise to the quantum to be paid under the policy liability being otherwise admitted such difference shall independently be referred to the decision of a sole arbitrator to be appointed in writing by the parties to if they cannot agree upon a single arbitrator within 30 days of any party invoking arbitration the shall be referred to a panel of three arbitrators ,comprising of two arbitrators ,one to be appointed by each of the parties to the dispute / difference and the third arbitrator to be appointed by such two arbitrators and arbitration shall be conducted under and in accordance with the provisions of the Arbitration and Conciliation Act, 1996, as amended by Arbitration and Reconciliation ( Amendment ) Act, 2015 ( No.3 of 2016 ).

It is clearly agreed and understood that no difference or dispute shall be referable to arbitration as herein before provided, if the Company has disputed or not accepted liability under or in respect of this policy.

It is hereby expressly agreed and declared that it shall be a condition precedent to any right of action or suit upon this policy that award by such arbitrator / arbitrators of the amount of the loss or damage shall be first obtained.

It is also hereby further expressly agreed and declared that if the Company shall disclaim liability to the insured for any claim hereunder and such claim shall not within 12 calendar months from the date of such disclaimer have been made the subject matter of a suit in a Court of Law, the claim shall for all purposes be deemed to have been abandoned and shall not thereafter be recoverable hereunder.

12. FREE LOOK PERIOD
i) The free look period shall be applicable at the inception of the policy and

(1)The insured will be allowed a period of at least 15 days from the date of receipt of the policy to review the terms and conditions of the policy and to return the same if not acceptable.

(2) If the insured has not made any claim during the free look period, the insured shall be entitled to—(a) A refund of the premium paid less any expenses incurred by the insurer on medical examination of the insured persons and the stamp duty charges or; (b) where the risk has already commenced and the option of return of the policy is exercised by the policyholder, a deduction towards the proportionate risk premium for period on cover or; (c)Where only a part of the insurance coverage has commenced, such proportionate premium commensurate with the insurance coverage during such period; (d) In respect of unit linked policy, in addition to the above deductions, the insurer shall also be entitled to repurchase the unit at the price of the units as on the date of the return of the policy.

13. GRIEVANCE REDRESSAL
In case of any grievance the insured person may contact the company through

i.         Website: www.newindia.co.in ii. Toll free: 1800 209 1415 iii.        E-mail: As stated in the policy schedule iv.         Fax : As stated in the policy schedule

v. Courier: As stated in the policy schedule

Insured person may also approach the grievance cell at any of the company's branches with the details of grievance.

If Insured person is not satisfied with the redressal of grievance through one of the above methods, insured person may contact the grievance officer at New India Head Office.

For updated details of grievance officer, kindly refer the link at www.newindia.co.in

Insurance Ombudsman —The insured person may also approach the office of Insurance Ombudsman of the respective area/region for redressal of grievance. The contact details of the Insurance Ombudsman offices have been provided as Annexure-A. Insureds are advised to note the revised details of insurance ombudsman as and when amended as available in the website http://ecoi.co.in/ombudsman.html 

Annexure-A.

The contact details ofthe Insurance Ombudsman offices are as below-

Areas of Jurisdiction

Office of the Insurance Ombudsman

Gujarat , UT ofDadra and Nagar

Haveli, Daman and Diu

Office of the Insurance Ombudsman, Jeevan Prakash Building,6th floor,TilakMarg, Relief Road,

Ahmedabad — 380001.

Tel.: 079-25501201 / 02 / 05/ 06

Email: bimalokpal.ahmedabad@ecoi.co.in

Karnataka

Office        of       the       Insurance       Ombudsman,

JeevanSoudhaBuilding,PID No. 57-27-N-19, Ground

Floor, 19/19, 24th Main Road,JP Nagar, 1st Phase, Bengaluru 560 078.

Tel.:        080                 26652048            / 26652049

Email: bimalokpal.bengaluru@ecoi.co.in
"""
policy_7_text = """Portable Equipment Insurance
Portable Equipment Insurance





Product Highlights

Highlights

Policy provides cover for the insured against accidental loss or damage to the portable equipment, whilst in the custody of the insured.

Scope

Risk covered:

1.  Fire & allied perils

2.  Terrorism

3.  Act of god perils

4. 

 
 	

Burglary

5.  Theft

6.  Robbery

7.  Accidental external means

Exclusions

1.   Loss due to war & war-like operations.

2.   Loss due to nuclear reaction, nuclear radiation or radioactive contamination from any source whatsoever.

3.   Loss due to overload, experiments or tests requiring the imposition of abnormal conditions.

4.   Loss due to gradually developing flaws, defects, cracks or partial fractures in any part not necessitating immediate stoppage.

5.   Loss due to wear & tear.

6.   Loss due to willful act or willful neglect or gross negligence.

7.   Loss due to faults or defects existing at the time of commencement of this Insurance.

8.   Loss due to consequential loss incurred by the Insured or legal liability of any kind.

9.   Loss due to breakage, cracking or scratching of crockery, glass cameras, binoculars, lenses, sculptures, curios, pictures, musical instruments, sports gear and similar articles of brittle or fragile nature, unless caused by fire or accident to the means of conveyance.

10. Loss to X-ray film or any electronic data storage media, data/records or similar non-tangible items.

11. Loss to articles/items of consumables in nature.

12. Theft from unattended vehicle.

Note:Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.



Process







Claims Process:

1. Immediate notification of the loss to be given to the Insurer.

2. Take all reasonable steps within his power to minimize the extent of the loss or damage.

3. Submission of claim form giving description of loss and supporting documents.

4. Preserve the damaged or defective parts and make them available for inspection by an official or surveyor of the company.

5. Immediate intimation to the Police and filing a FIR following a theft.

6. Furnish all such information and documentary evidence as the Company may require.



FAQs:

1.  Who can take the policy?

Owner of the equipment i.e. individual or organization can take this policy.

2.  How to choose the sum insured for Portable equipment policy?

Sum Insured shall be equal to the cost of the replacement of the property by property of the same kind and same capacity which shall mean its replacement cost including freight, dues and customs duties, if any.

3.  Does Portable equipment insurance cover electrical items?

Yes, equipment that can be moved from one place to another can be covered under this policy.

4.  Can Laptop be covered under this policy?

http://bookstack.zubera.one/link/332#bkmrk-yes%2C-only-for-accomp
 
Yes, only for accompanied laptops."""
policy_8_text = """Bankers Indemnity Insurance


Bankers Indemnity Insurance



Package policy designed specially to cover the risks related to banking sector. A single policy covering all branches in India of the particular bank.

Highlights
A package policy designed specially to cover the risks related to banking sector. A single policy covering all branches in India of the particular bank.

Retroactive period facility available whereby losses discovered during policy period due to an incident occurring in earlier period but after inception of first policy, also become payable, provided the policy has been continuously renewed with us without break.

Discount in premium available for banks having less than 500 branches.

Scope

The policy comprises of following 7 sections .

1. On Premises : Covers money and/or securities belonging to, or in the custody of bank, whilst on their own premises or on the premises of their bankers, against loss or destruction by Firer Riot & Strike, Malicious damage, terrorist act, burglary ,theft ,robbery or hold-up.

2. In Transit : Covers money and/or securities if they are lost ,stolen, mislaid, misappropriated or made away with, whilst in transit in the hands of its employees whether by negligence or fraud of the employees.

3. Forgery or Alteration : Covers losses suffered as a result of payment of bogus, fictious, forged cheques or drafts as also forged endorsements on genuine cheques or drafts or FDRs.

4. Dishonesty : Covers loss of money and/or securities suffered due to dishonest or criminal act of its employees.

5. Hypothecated Goods : Covers losses suffered due to fraudulent or dishonest act of employees in respect of goods or commodities pledged or hypothecated to the insured bank and under its control.

6. Registered Postal Service : Covers loss of registered postal sending by robbery,theft or any other cause not specifically excluded, provided that each post parcel shall be insured with the post office.

7. Appraisers : Covers loss due to infidelity or criminal act on the part of appraisers, provided that such appraisers are on the bank's approved list.

8. Janata Agents : Covers loss due to infidelity of criminal acts on the part of Janata Agents, Chhoti Bachat Yojana Agents/Pygmie Collectors.

Add on covers

The following additional perils can be covered on payment of an additional premium .

1. Losses due to flood, inundation, hurricane, typhoon, storm, tempest, tornado and cyclone.

2. Losses due to earthquake - Fire & Shock

3. Additional sum insured can be opted for under Section A & B.

Who can take the policy?

Any banking company as defined under various Banking Acts like Banking Regulation Act 1945, State Bank of India Act 1955 etc.

How to select the sum insured?

The proposer has to select a basic sum insured which will apply to Sections A to E of the policies. This sum insured should represent the maximum amount of loss which could be suffered by the bank due to any single incident covered under Sections A to E. The sum insured under Section F,G&H is fixed at a percentage of the basic sum insured.

In addition to the basic sum insured , an additional sum insured can be opted under Section A and/or B on payment of additional premium.

How to claim?

In case of discovery of any loss falling under the scope of the policy, the following steps should be taken:

1. Inform insurance co. by phone and/or fax/letter.

2. In case of burglary/robbery/theft/hold-up etc. inform police and get FIR registered.

3. In case of dishonest act of employee, inform police and initiate departmental enquiry.

4. Submit claim form and relevant documents to substantiate loss to the surveyor appointed by the insurance company.

5. Take reasonable steps to prevent further loss due to the same reason.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

Product FAQ







 

FAQ

 

1. Who can take this policy?

Any nationalized, Scheduled or Cooperative Bank can take this policy.

2. What should be the policy period of the policy ?

The policy should commence and end as per financial year of the bank .

3. What is the cover under the policy ?

The policy covers Moneys / Securities held by the bank whilst on Premises being lost, destroyed or otherwise made away with by Fire, Riot and Strike, Burglary, Housebreaking , theft, Robbery, Hold-up, Dishonesty of Employees and Loss in Transit.
"""
policy_9_text = """Bhagyashree Insurance
Bhagyashree Insurance

Highlights
The scheme is intended to provide insurance cover to ONE girl child in a family who loses either the father or the mother due to accidental death. The insurance cover is available on 24 hour risk basis. Incase of death of parents, the company deposits a sum of INR 25000/- in the name of the girl child mentioned in the schedule of the policy with a financial institution named in the schedule. The premium is INR 15/- per girl child per year. Group discount is also provided.

Scope

The policy covers death of one or both parents of the girl by accident caused by external, violent and visible means would include death or permanent total disablement arising out of or traceable to slipping, falling from the mountain, insect bites, snakes and animals bite, drowning, washing away in floods, landslide, rockslide, earthquake, cyclone and other commotions or nature and/or calamities, murder or terrorist activities In case of women it also includes death and PTD due to surgical operations such as sterilisation, ceasarean, hysterectomy i.e. removal or uterus and removal or breasts due to cancer operations, death at the time of child birth provided that such death occurs during the surgical operation in hospital/nursing home or whilst being in the hospital/nursing home after such surgery convalescene. However not beyond a period of 7 days from the date of surgical operations.

Eligibility
This scheme is applicable to girl children in the age group of 0 to 18 years, whose parents' age does not exceed 60 years.

This scheme is applicable to girl children in the age group of 0 to 18 years, whose parents' age does not exceed 60 years.

Major Exclusions

Pre existing disability, death, injury or disablement arising from or traceable to whilst under the influence of intoxication, liquor and drugs, Death caused by earthquake or other convulsions of nature, suicide and intentional self injury. Death or injury directly or indirectly caused by insanity, nuclear weapons etc.

http://bookstack.zubera.one/link/352#bkmrk-note%3A-policy-details
 
Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details."""
policy_10_text = """Burglary Insurance
Burglary Insurance

Policy is designed to cover business premises only like godown, factory, office etc.

Highlights

Policy is designed to cover business premises only like godown, factory, office etc.

There are three types of policies available :-

•  Full Value Insurance: The policy must be effected for the full value of the property to be insured.

•  First Loss Insurance: In the event of improbability of total loss, proposer can opt for a percentage of total stocks to be insured

•  Stock Declaration Policies: These policies are given where large stocks frequently fluctuate in quantity during the year. The sum insured is fixed at the maximum value of stocks which the insured anticipates he will hold at any one time. A deposit premium of 100% of the annual premium will be paid at the beginning of the insurance. Monthly declarations of value are to be sent to the company and the 'deposit' premium will be adjusted at the end of the policy period based upon the average of the monthly declarations

Scope

1. Loss or damage to the property insured by theft following upon actual, forcible and violent entry into the premises.

2. Damage to the premises following upon entry as above or any attempt thereat

The indemnity provided is to the extent of the intrinsic value of the property so lost or damaged, subject to the limit of the sum insured.

Exclusions

The company shall not be liable in respect of :

1. Gold, watches, jewellery, precious stones, plans, designs, money, business books etc. unless specifically insured.

2. Loss or damage where any insured or member of the insured's household or of his business staff is concerned in the actual theft or damage.

The policy shall cease to attach:

1. If the premises are left uninhabited for 7 or more consecutive days and nights.

2. In the event of material alterations to the premises whereby the risk is increased.

3. If the insurable interests has passed from the insured otherwise by will or operation of law

In event of claim

1. The insured should give immediate notice to the police and also to the company and within 14 days submit to the company his claim in respect of loss or damage sustained.

2. The insured should also tender to the company all reasonable information, assistance and proofs in connection with any claim here under.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

Claim

Process





1. Immediate notification of the loss to be given to Insurer.

2. Immediate intimation to the Police and filing a FIR

3. Submission of claim form giving description of loss and supporting documents.



1. What is the cover under this policy ?

Loss or damage to the property of a business organization by theft following upon forcible , violent entry into or exit from the premises.

2. Does it cover money in the premises ?

Money can be covered by making specific mention while taking the policy.

3. Whether documents can be covered ?

Yes, by including it specifically while taking the policy.

4. What if full value of the property has not been covered ? The claim would be reduced proportionately.

5. Whether burglary or theft during riots is covered ? Yes. The extension can be taken under the policy."""
policy_11_text = """E Flight Coupon Insurance
E Flight Coupon Insurance

Product Highlights

Product Highlights

Scope:

Passengers flight insurance coupons cover death, permanent disability, and any bodily injury caused by violent, accidental, external, or visible means whilst in or entering into or descending from any aircraft owned and/or operated by a regular airline over a scheduled route by which the insured is traveling as a passenger during the flights specified. The scale of benefits is shown in the standard policy form prescribed for this class of insurance.

Exclusions:

•   War and allied perils.

•   If arising whilst the insured is under the influence of intoxicants or is suffering from lunacy or insanity.

•   If arising from disobedience of instruction of aircraft crew, aircraft owners or operators, or their agents or servants.

•   Accidental death of the insured shall not be presumed by reason of his disappearance.

•   The insurance is not valid if the insured is under 12 years of age or is over 70 years.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.



Claims Process:

1    . Immediate notification of the loss to be given to Insurer.

2    Submission of claim form giving description of loss and supporting documents.

http://bookstack.zubera.one/link/350#bkmrk-3%C2%A0%C2%A0%C2%A0-furnish-all-suc
 
3    Furnish all such information and documentary evidence as the Company may require"""
policy_12_text = """Event Cancellation Insurance
Event Cancellation Insurance

Product Highlights

Product Highlights

Scope

The insurance is to indemnify the insured for their ascertained net loss should any insured event(s) specified in the schedule be necessarily cancelled, abandoned, postponed, interrupted, or relocated, in whole or in part, which necessary cancellation, abandonment, postponement, interruption, or relocation is the sole and direct result of any cause beyond the control of the Assured and the participants therein.

Event means SOCIAL EVENTS.

Coverages

* Section l: Cancellation of event due to Fire & Allied perils for Sum Insured.

* Section Il: Public Liability Insurance — Indemnity limit AOA/AOY (IF OPTED).

Period of Insurance

As per the requirement.

Exclusions

* Non-appearance of individual members, officials, speakers, teams, players, performers, performing groups, participants, exhibitors, or guests.

* Duty of care: The assured's lack of care, diligence, or prudent behavior, resulting in increased risk and/or likelihood of loss.

* Any contractual dispute or breach by the assured & such other.

* Adverse weather in respect of outdoor events.

* Unavailability of venues.

* Any event in open or under canvas or in a temporary structure unless agreed by the Insurer.

* Civil commotion.

* Seepage and/or pollution and/or contamination unless discovered during the policy period and is a direct cause of loss.

* Withdrawal, insufficiency, or lack of finance howsoever caused.

* Financial failure of any venture.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

Claim Process

Claims Process:

1 . Immediate notification of the loss to be given to Insurer.

2. Submission of claim form giving description of loss and supporting documents.

http://bookstack.zubera.one/link/349#bkmrk-3.-furnish-all-such-
 
3. Furnish all such information and documentary evidence as the Company may require."""
policy_13_text = """Fidelity Guarantee Insurance


Highlights

The policy covers the employer in respect of any direct financial loss which he may suffer as a result of employees dishonesty.

Scope

The Company agrees to indemnify the insured against a direct pecuniary loss sustained by reason of any act of fraud/dishonesty committed

1. On or after the date of commencement of this policy

2. During uninterrupted service with the Insured and discovered during the continuance of this policy or within twelve calendar months of the expiration thereof

3. In the case of death, dismissal or retirement of the Employee with twelve calendar months of such death, dismissal or retirement whichever of these events shall first happen.

Conditions

Fidelity Guarantee Insurance

The liability of the Company shall not exceed

1. In respect of any employee the sum insured stated against his name or as declared herein.

2. In respect of all claims under this policy, the total sum insured.

2. If this policy shall be continued in force for more than one period of indemnity or if any liability shall exist on the part of the Company under this Policy and also under any other Policy in respect of fraud or dishonesty of the employee, the liability of the Company hereunder shall not be accumulated or increased thereby but the aggregate liability of the Company during any number of periods of indemnity and for any number of acts of fraud or dishonesty committed by the employee shall not exceed the sum insured hereunder or the sum insured under any other such policy as aforesaid whichever is greater.

3. The Company shall not be liable to pay more than one claim in respect of the action of any one employee.

Exceptions

The Company shall not be liable in respect of losses arising elsewhere than in India.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

1.  Immediate notification of the loss to be given to Insurer.

2.  Immediate intimation to the Police and filing a FIR

3.  Submission of claim form giving description of loss and supporting documents.

4.  Take all necessaty practicable steps to recover the property lost, apprehend the culprits and take appropriate action against them.

5.  Conduct a thorough internal inquiry and make the findings available to the insurers.

6.  Take all practicable steps to withhold any monies due to the defaulting employees.

Product FAQ







1. What is Fidelity Insurance ?

The policy covers direct financial loss incurred due to infidelity of employees.

2. Does it cover only financial loss ?

Yes. The indemnity is for the amount of money lost or for the value of the stock lost.

3. Does the policy covers dishonesty of employees of contractors or sub-contractors also ? The extension can be given to cover employees of contractors and sub-contractors.

4.

 
 	

Whether bigger organizations are more prone to this type of loss or smaller ones ?

Smaller organizations are more prone to such losses specially where employee changes are more frequent.

5. Is it necessary to give names of all employees to be covered under the policy ?

The policy can be issued both on named and unnamed basis. The policy can also be issued covering employees by their designations f categories.

6. Is it required to give amounts of insurance for each employee ?

separate amounts can be given if number of employees is small. For a large no. of employees, a common overall sum insured can be selected with respective limits for different categories of employees.

Fidelity Guarantee Policy :-

Dos & Don'ts:

 Recruitment of employees has to be done with sufficient background checks

 Proper procedural checks and balances.

 Regu lar audit and inspections are a must.

 Proper and acceptable accounting standards.

 Dual control and job rotations at regular intervals are helpful in preventing frauds.

 To ensure Employee satisfaction which in turn acts as loss avoidance measure.  An employee found guilty / suspect, must not be employed again."""
policy_14_text = """JANATA PERSONAL ACCIDENT POLICY
JANATA PERSONAL ACCIDENT POLICY

This document provides only key information about your policy. Please refer to the policy document for detailed terms and conditions.

Sl

No

Title

Description

(Please refer to applicable Policy Clause Number in next column)

Policy /

Clause

Number

1

Product Name

Janata Personal Accident

 

2

Unique Identification Number (I-JIN) allotted by

IRDAI

UIN No. IRDA/NL-HLT/NlA/P-P/V.l/55/14-15

 

3

Structure

Fixed Benefit

Policy clause pg no. 1

4

Interests Insured

Accidental Death and bodily injury

Policy clause pg no. 1

5

Sum Insured

Rs. 25,000/ Rs. 50,000/ Rs. 75,000/ Rs.

 

6

Policy Coverage

If the Insured shall sustain any bodily injury resulting solely and directly from Accident caused by outward, violent and visible means then the Company shall pay to the insured the sum hereinafter set forth:

a)                  If such injury shall within twelve Calendar months of its occurrence be the sole and direct cause of the death of an insured person the Capital Sum Insured in the Schedule hereto.

b)                 If such injury shall within Twelve calendar months of its occurrence be the sole and direct cause of the total and irrecoverable loss of sight of both eyes or both hands or both feet or of the actual loss of one eye and such loss of one of the two entire hands or two entire feet, or of one entire hand and one entire foot, or of such loss of sight of one eye and such loss of one entire hand of such loss of one entire foot of an Insured person the Capital Sum Insured in the Schedule hereto.

c)                  If such injury shall within twelve calendar months of its occurrence be the sole and direct cause of the total and irrecoverable loss of sight of one eye, or of the actual loss of one entire hand or one entire foot of an Insured person, fifty per cent (50%) of the Capital Sum Insured stated in the Schedule hereto.

Policy clause pg no. 1

 

 

 

d) If such injury shall as a direct consequence thereof immediately, permanently, totally and absolutely disable an Insured person from engaging in being occupied with or giving attention to paid employment or occupation of any description whatsoever, the capital sum insured stated in the Schedule hereto.

 

7

Add-on Cover

Nil

 

8

Loss Participation

Nil

 

9

Exclusions

he Company shall not be liable under this Policy for:

1.               Compensation under more than one of the subclauses (a), (b), (c) or (d) in respect of same injury or disablement.

2.               Payment of compensation in respect of injury or disablement directly or indirectly arising out of or contributed to by or traceable to any disability existing on he date of issue of this Policy.

3.               Payment of compensation in respect of death, injury or disablement of the Insured from (a) intentional elf injury, suicide or attempted suicide. (b) whilst under he influence of intoxicating liquor or drug. (c) directly o indirectly caused by insanity. (d) arising or resulting from he insured committing any breach of the law with criminal intent.

4.               Payment of compensation in respect of death, injury or disablement of the Insured from (a) due to or arising out of or directly or indirectly connected with or raceable to war, invasion, act of foreign enemy, hostilities (whether war be declared or not) Civil war, rebellion, revolution insurrection, mutiny, military or usurped power, eizure, capture, arrests, restraints and detainments of all kings, princes and people of whatsoever nation, condition or quality.

5.               Payment of compensation in respect of death of or bodily injury to the Insured directly or indirectly caused by or contributed to by or arising from or traceable o ionizing radiation or contamination by radioactivity from any source whatsoever, or from nuclear weapons material.

Policy clause pg no. 2


Provided also that due observance and fulfillment of the terms and conditions of this Policy (which conditions and all endorsements hereon are to be read as part of this Policy) shall so far as they relate to anything to be done or not to be done by the Insured be a condition precedent to any liability of the Company under this Policy.


10.

Special Conditions and Warranties (if any)

1. In the case of a claim by death or permanent total disablement all sums will be payable only on the delivery of this Policy canceled and discharged

2. Any other, as specified, in the policy schedule.

Policy clause pg no. 2

11.

Admissibility of Claim

1. Upon the happening of any event which may give rise to a claim under this policy the Insured shall forthwith give notice thereof to the Company. Unless reasonable cause is shown, the Insured should within one calendar month after the event which may give rise to a claim under the policy, give written notice to the Company with full particulars of the claim.

2.  Proof satisfactory to the Company shall be furnished of all matters upon which a claim is based. Any medical or other agent of the Company shall be allowed to examine the insured person on the occasion of any alleged injury or complement when and so often as the same may reasonably be required on behalf of the Company and in the event of death, to make a postmortem examination of the body of the insured and such evidence as the Company may from time to time require (including a post-mortem examination, if necessary) shall be furnished within the space of fourteen days after demand in writing. Provided that in the case of a claim by death or permanent total disablement all sums will be payable only on the delivery of this Policy canceled and discharged.

Policy clause pg no. 2

12.

Policy Servicing -

Claim Intimation and Processing

1800-209-1415

Website-https://www.newindia.co.in Policy issuing office

 

13.

Grievance Redressal and

Policyholders Protection

Details of Grievance redressal officer- available at NIA website:https://www.newindia.co.in/portal/readMor e/Grievances

IRDAI Integrated Grievance Management System — https//igms.irda.gov.in/l

Insurance Ombudsman — The contact details of the Insurance Ombudsman offices has been provided in the website

14.

Obligations of the Policyholder

To disclose all information correctly sought by the insurer at time of filling the proposal form

In case of any change / modification / addition to the already declared information the same shall be brought to the notice of the Insurer immediately

Non-disclosure of material information may affect the claim settlement.


Declaration by the Policyholder

I have read the above and confirm having noted the details.

Place:

   Date:                                                                            (Signature of the Policyholder)



Note:

Insurer shall provide web-link where the product related documents including the

Customer Information sheet are available on the website of the Insurer.

ii.           Insurer to take confirmation of the Policyholder regarding receiving of the Customer

Information Sheet.

http://bookstack.zubera.one/link/345#bkmrk-iii.%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0-the-inf
 
iii.         The information must be read in conjunction with the product brochure and policy document. In case of any conflict between the CIS and the policy document, the terms and conditions mentioned in the policy document shall prevail."""
policy_15_text = """Mahila Udyam Bima


New India Mahila Udyam Bima is exclusively designed to provide protection to women entrepreneurs running small businesses having an asset value not exceeding 5 Cr.

Industries covered include:

* Weaving and other cottage industries

* Beauty parlors, salons

* F ood join ts/eateries

* Event management companies

* Training institutes

* Boutiques

Scope of Covers

A. Compulsory Coverages

1. Fire & Allied perils— Covers building, plant and machinery, stocks, furniture, and fixtures.

2. Loss of Profits— Covers losses due to business interruption.

3. Burglary— Covers burglary and housebreaking due to violent and forcible entry/exit.

4. Personal Accident For SME owner— Covers bodily injury caused by external means.

B. Optional Covers

•   Public liability— Covers legal liability to third parties.

•   Personal Accident for Employees— Covers bodily injury.

•   Employee compensation- Covers up to 20 employees.

Exclusions

•   Loss due to war, civil commotion, wear & tear

•   Theft unless caused by violent means

•   Consequential losses

Claims

1. Immediate notice of claim to insurer.

2. Lodge a police complaint/FlR if applicable.

3. Provide death certificate/post-mortem report if applicable.

4. Maintain and provide invoices, receipts, records of payment, etc.

Cancellation

*  Insured can cancel anytime with a pro-rata premium refund.

*  No refund if claims have been notified.

*  Insurer can cancel only for non-payment, misrepresentation, or fraud.

Special Features

*  Discount up to 20% on total premium.

*  Additional 2.5% discount if renewable power sources are used.

Product FAQ



FAQ

 



FAQs

New India Mahila Udyam Bima ( UIN NO. IRDAN190RPMS0034V01202425)

1. Who can take the policy?

• Women Entrepreneurs running Micro and small industries having an asset size/value not exceeding 5 crores

2. Is MSME registration certificate required?

• Not compulsory

3. Can it be given to service organisations?

• Yes

4. In a partnership firm, if some partners are male members can this policy be issued?

• Yes. There should be at least one woman partner' Discount up to 20% can be given as per underwriters assessment of risk

5. Can it be issued if the women entrepreneur is working from their residential premises?

• Yes. Only to the extent of assets used for business. Buildings be included in the sum insured if it is in the name of the woman entrepreneur.

6. Can it be given to the agricultural sector?

• The policy cannot be given where crop/grain and other agricultural produce are involved

7. Can theft and RSMD be covered under the burglary section?

• NO

8. Cannot extend or modify the sections now?

• As the policy has already been filed, any modifications in the existing structure are not possible

9. Sum insured under Section 1

• Please take bifurcation of sum insured for buildings and contents. We will enable the same in the system soon.

10. LOP section — For Manufacturing firms

1. Sum insured should be equal to the value of contents declared in Section 1.

2. Indemnity period will be max 365 days.

3. The indemnity would be No of FULL WORKING DAYS lost due to operation of insured peril under section 1 / No of working days in the indemnity period *(Sum insured in Section B or Actuals whichever is lower)

11. LOP section - For service sectors

1. Indemnity period will be max 365 days.

2. The indemnity would be — No of FULL WORKING DAYS lost due to operation of insured peril under section 1 / No of working days in the indemnity period *(Sum insured in Section B or Actual loss or revenue/fees whichever is less)

12. Burglary is on 50% first loss basis on the sum insured of contents declared for section 1 of the policy.

13. What is the eligibility for providing a discount for use of renewable resources?

• Use of renewable resources for business purposes. Not necessarily to the extent of 100%

14. Can Medical expenses be included in WC section?

• Yes. Medical expenses covered up to Rs.50,000.

15. Can this policy be issued to women directors on board?

 NO"""
policy_16_text = """Money Insurance


Highlights

Money Insurance policy provides cover for loss of money in transit between the insured's premises and bank or post office,or other specified places occasioned by robbery, theft or any other fortuitous cause.

The policy also cover loss by burglary or housebreaking whilst money is retained at Insured's premises in safe(s) or strong room.

Scope of Cover

Section l: Covers money in transit under the following heads: Cash, Bank Drafts, Currency Notes, Treasury Notes, Cheques, Postal Orders and current Postage Stamps.

Section Il: Covers money in safe / on premises

Basis of Sum Insured

Two amounts are specified in the policy:

Limits of liability for any one loss (i.e. maximum liability of the Company)

Estimated amount in transit during the year for the purpose of premium computation.

Extensions

This policy can be extended to include the risk of infidelity of the employees, terrorism and disbursement risk.

Exclusions

1. Shortage due to error or omission

2. Losses due to the fraud/dishonesty of the employee of the insured.

3. Losses which are covered by other policies

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.



Claims Process:

1. Immediate notification of the loss to be given to Insurer.

2. Take all practicable steps to discover the guilty person or persons and to recover the cash lost.

3. Immediate intimation to the Police and filing a FIR.

4. Submission of claim form giving description of loss and supporting documents.

5. Insured shall furnish all explanations, vouchers, proof of ownership and other evidence to substantiate the claim and the Company may, if it deems necessary, require corroborative evidence of the statements of the Insured or any of Insured's family members or employee/s.

FAQs:

1. Who can take the policy?

Any industrial establishment or company that deals with or draws money daily for their day-to-day transactions can take the policy.

2. What is covered under the policy?

Money Insurance policy provides cover for loss of money in transit between the insured's premises and bank or post office, or other specified places occasioned by robbery, theft, or any other fortuitous cause. The policy also covers loss by burglary or housebreaking whilst money is retained at the Insured's premises in safe(s) or strong room.

3. What would the definition of the money?

Money shall mean and include cash, bank drafts, currency notes, cheques, postal orders, money orders, and current postage stamps.

4. What would be the sum insured under the policy?


 
 	

Sum insured shall be estimated annual turnover of the company, cash in transit, estimated single carrying cash limit, and cash stored in safe.

5. Can terrorism risk be covered under the policy?

Yes, terrorism risk can be covered with additional premium.

6. What is not covered under the policy?

The common exclusions under the policy are as below:

 Loss due to flood, cyclone, earthquake & other convulsions of nature.

 Loss due to war & war-like operations.

 Shortage due to error or omission.

 Loss of money entrusted to any person other than the Insured or an authorized employee of the Insured.

 Loss occurring on the premises after business hours, unless the money is in a locked safe or strong room. o Theft from unattended vehicle.

http://bookstack.zubera.one/link/343#bkmrk-%C2%A0consequential-loss-
 
 Consequential loss or legal liability of any kind."""
policy_17_text = """Neon Sign Insurance


Highlights

Insurance in respect of loss or damage to the neon sign installation.

Scope

Covers loss or damage to the neon sign installation by

1.  Accidental external means

2.  Fire, lightning, external explosion and theft

Exclusion

1.  Fusing or burning out of any bulbs/tubes arising from short circuiting or arcing or any other mechanical or electrical defect or breakdown

2.  Repair, cleaning, removal or erection, wear and tear, depreciation or deterioration

3.  Damage to tubes unless the glass in fractured

4.  Over running, over heating or strain

5.  Atmospheric conditions

6.  War and kindred perils

7.  Natural risks

Neon Sign

Insurance

8.  Consequential loss

Special Condition

Insured neon sign must be examined and inspected at regular intervals of not longer than 6 months by a qualified electrician or engineer.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.



Claims Process:

1. Immediate notification of the loss to be given to Insurer.

2. Submission of claim form giving description of loss and supporting documents.

3. Furnish all such information and documentary evidences as the Company may require.

4. Insured shall take all reasonable precautions to protect the neon sign insured hereunder.

5. In case of theft, report to police or FIR is necessary.

FAQs:

1.  Who can take this policy?

Anyone can take the policy for the neon sign having substantial value.

2.  What is covered under the policy?

The policy covers loss or damage to the neon sign due to accidental external means, fire, lightning, external explosion, theft, and malicious acts.

3.  What is meant by neon sign?

Neon sign hoardings are used for advertisement displays located in public places such as neon signs, LED signs, or any digital signs.

4.  What is the maximum indemnity in the policy?

Product FAQ



FAQ

 

The maximum indemnity is the value of the neon sign at the time of occurrence of loss or the insured's estimate of value mentioned in the policy, whichever is less.

5.  What is not covered under the policy?

The following common losses are not covered under the policy:

 Fusing or burning out of any bulbs/tubes arising from short circuiting or arcing, or any other mechanical or electrical defect or breakdown.

 Repair, cleaning, removal or erection, wear and tear, depreciation, or deterioration.

 Damage to tubes unless the glass is fractured.

 Overrunning, overheating, or strain.

 Atmospheric conditions.  War perils.

o Natural risks.

http://bookstack.zubera.one/link/340#bkmrk-%C2%A0consequential-losse
 
 Consequential losses."""
policy_18_text = """NEW INDIA PENSION PROTECT PERSONAL ACCIDENT POLICY
NEW INDIA PENSION PROTECT PERSONAL ACCIDENT POLICY

(UIN No: NIAPAlP23064V012223)

1. PREAMBLE
This Policy is a contract of insurance issued by The New India Assurance (hereinafter called the 'Company') to the proposer mentioned in the schedule (hereinafter called the 'Insured') to cover the person(s) named in the schedule (hereinafter called the 'Insured Persons The policy is based on the statements and declaration provided in the proposal Form by the proposer and is subject to receipt of the requisite premium.

2. OPERATIVE CLAUSE
Any amount payable under the policy shall be subject to the terms of coverage, exclusions, conditions and definitions contained herein. Maximum liability of the Company under all such Claims during each Policy Year shall be the Sum Insured specified in the Schedule.

3.       POLICY COVERAGE Basic Cover:

Accidental Death-Full Sum Insured

Built In Covers:

Carriage of Dead Body— I % of CSI maximum Rs. 2500/Funeral Expenses — Rs. 2500/-

4.       DEFINITIONS:

4.1  ACCIDENT

An accident is a sudden, unforeseen, involuntary event caused by external, visible and violent means.

4.2  INJURY

Injury means accidental physical bodily harm excluding illness or disease solely and directly caused by external, violent and visible and evident means which is verified and certified by a medical practitioner.

4.3  SUM INSURED

Maximum sum insured allowable is 72 times of Monthly Pension Lost (as per the Pension Scheme of the respective employer / Annuity Plan purchased by the Employer / Annuity Plan purchased by the retiree ) in the event of Death of the Pensioner.

Pension Lost implies the difference between the derived pension as on the date of commencement of the policy period and the family pension available to such dependent family members as would become recipient of the family pension as recorded in the Pension Scheme / Annuity Plan after the death of the pensioner.

4.4  FAMILY PENSION

Family Pension is defined as a regular monthly amount which an employer / insurer of the Annuity Plan would pay to a person who belongs to the family of the employee in the event of the Death of the employee.

4.5  AGE

Entry Age - From the time of becoming a Pensioner, but not less than 35 years and up to age of 70 years. Age means age ofthe Insured Person on the last birthday as on the date ofcommencement of the policy.

(Note: On completion of 65 years of age the acceptance of new proposals would be subject to submission of satisfactory physical fitness certificate from family doctor / medical practitioner and premium loading @ 2 % every year.)

4.6  POLICY PERIOD means period of one year for which the policy is issued.

4.7  GRACE PERIOD

Grace period (30 days or as amended by IRDA from time to time) means the specified period of time immediately following the premium due date during which a payment can be made to renew or continue a policy in force without loss of continuity benefits. Coverage is not available for the period for which no premium is received by the Company.

4.8  RENEWAL

Renewal defines the terms on which the contract of insurance can be renewed on mutual consent with the provision of grace period for treating the renewal continuous.

4.9  BENEFIT PAYABLE -

If at anytime during the currency of this policy, the Insured person shall die

(a) resulting solely and directly from Accident , then the Company shall pay to such dependent family members as would become recipient of the family pension as recorded in the Pension Scheme / Annuity Plan purchased by the Retiree which is the basis of this policy contract. (b) Provided such death shall have occurred within Twelve months of the date of such Accident.

Basic Cover:

Accidental Death-Full Sum Insured

Built In Covers
Carriage of Dead Body— 1 % of CSI maximum Rs. 2500/Funeral Expenses — Rs. 2500

5. EXCLUSIONS

The Company shall not be liable to make any payment under this policy in respect of any Benefit for Death of the Insured from

(a)    From intentional self —injury

(b)    From suicide

(c)    From voluntary self exposure to sports / hazardous activities / adventure sports /Adventure activities

(d)    Any claim arising due to illness

(e)    Whilst under the influence of intoxicating liquor or drugs

(f)     Whilst engaging in Aviation or Ballooning ,whilst mounting into or dismounting from or travelling in any balloon or aircraft other than as a passenger (fare paying or otherwise) in any licensed standard type of aircraft anywhere in the world

'Standard Type of Aircraft' means any aircraft duly licensed by appropriate authority to carry passengers (for hire or otherwise) irrespective of whether such an aircraft is privately owned OR chartered OR operated by a regular airline OR whether such an aircraft has a single engine or multi-engine.

(g)    Directly or indirectly caused by any disease, veneral disease / diseases or insanity"

(h)    Circumcision or Strictures or Vaccination or Inoculation or change of life or beauty treatment of any description or dental or eye treatment or dissipation or nervous breakdown (which expression shall also cover general debility (rundown conditions and general overhaul ) or veneral disease or intemperance .

(i)     Arising or resulting from the Insured committing any breach of law with criminal intent.

(j)     Arising out of directly or indirectly connected with or traceable to —War,Invasion,Act of foreign enemy, Hostilities (whether war be declared or not ) , Civil war, Rebellion, Insurrection, Mutiny,Military or Usurped power Seizure, Capture, Arrests, Restraints and Detainment by any kings, princes and people of whatever nation, condition or quality.

(k)    Any claim resulting or arising from or any consequential loss directly or indirectly caused by or contributed to or arising from:

A.   Ionizing radiation or contamination by radioactivity from any nuclear fuel or from any nuclear waste from the combustion of nuclear fuel or from any nuclear waste from combustion (including any self-sustaining process of nuclear fission) of nuclear fuel.

B.   Nuclear weapons material

C.   The radioactive, toxic, explosive or other hazardous properties of any explosive nuclear assembly or nuclear component thereof.

D.   Nuclear, chemical and biological terrorism

(l) Any loss arising out of the Insured Person's actual or attempted commission of or wilful participation in an illegal act or any violation or attempted violation of the law.

6.      CLAIMS PROCEDURE & CONDITIONS 

6.1.   Notification of claim:

i.        Intimation about an event or occurrence that may give rise to a claim under this policy must be given within 30 days of its happening.

ii.      Claims for insurance benefits must be submitted to the Company not later than one (1) month after transportation of the mortal remains/ burial in the event of Death.

Note:

1.    The Company will examine and relax the time limit mentioned herein above depending upon the merits of the case.

2.    Proof satisfactory to the Company shall be furnished of all matters upon which a claim is based. Any medical or other agent of the Company shall be allowed to examine or have post-mortem examination of the Insured, as may reasonably be required on behalf of the Company. Such evidence as the Company may from time to time require shall be furnished and the post-mortem examination report if necessary, be furnished within the space of fourteen days after the demand is raised in writing.

3.    In case of death of the insured person the policy automatically ceases to be operative, without any refund of premium under any circumstances.

4.    No sum payable shall ordinarily carry any interest. In case of any extra ordinary delay on the part of insurer ,such claims shall be paid by the insurer as specified in IRDA (Protection of Policyholder's Interest) regulations 2017 dated 22.06.2017

5.    The Company shall not be liable to make any payment under this policy in respect of any claim if such claim be in any manner fraudulent or supported or by any fraudulent statement or device, whether by the Insured or by any person on behalf of the Insured.

6.2.   Documents required for processing a claimBasic documents required for claims

i. Duly completed claim form ii. Photo Identity Proof of the insured person ili. Copy of FIR/ Panchnama / Police Inquest Report (wherever these reports are required as per the circumstance of the Accident) duly attested by the concerned Police Station iv. Copy of Medico Legal Certificate (wherever it is required as per the circumstance of the Accident) duly attested by the concerned Hospital

v. Death certificate; vi. Post Mortem Report (if conducted); vii. Identity proof of Nominee or Family Pension Recipient Original Succession

Certificate / Original Legal Heir Certificate or any other proof to the satisfaction of the Company for the purpose of a valid discharge in case nomination is not filed by deceased. Any other relevant document required by the Company for assessment of the claim viii. Any other relevant document required by the Company for assessment of the claim

6.3.   Payment of claim

All claims under the policy shall be payable in Indian currency only.

6.4.   Claim Settlement

i.               The Company shall settle or reject a claim, as the case may be, within 30 days from the date of receipt of last necessary document.

ii.              In case of delay in the payment of a claim, the Company shall be liable to pay interest to the policyholder from the date of receipt of last necessary document to the date of payment of claim at a rate 2% above the bank rate.

iii.            However, where the circumstances of a claim warrant an investigation in the opinion of the Company, it shall initiate and complete such investigation at the earliest, in any case not later than 30 days from the date of receipt of last necessary document. In such cases, the Company shall settle or reject the claim within 45 days from the date of receipt of last necessary document.

iv.             In case of delay beyond stipulated 45 days, the Company shall be liable to pay interest to the policyholder at a rate 2% above the bank rate from the date of receipt of last necessary document to the date of payment of claim.

(Explanation: "Bank rate" shall mean the rate fixed by the Reserve Bank of India (RBI) at the beginning ofthe Financial Year in which claim has fallen due)

7.      RENEWAL

The policy shall ordinarily be renewable except on grounds of fraud, misrepresentation by the insured person.

The Company shall endeavour to give notice for renewal. However, the Company is not under obligation to give any notice for renewal.

Request for renewal along with requisite premium shall be received by the Company before the end of the policy period.

At the end of the policy period, the policy shall terminate and can be renewed within the Grace period of 30 days to maintain continuity of benefits without break in policy. Coverage is not available during the grace period.

The cover for the Insured shall terminate immediately in the event of admissible claim and settlement of 100% Sum Insured under Accidental Death Coverage and no Renewal of contract will be permissible.

The Insured shall give immediate notice to the Company of any change in status of the pensioner or source of income of the insured person, other than the pension.

The Insured shall, on tendering any premium for the renewal of this policy, give notice in writing to the of any disease, physical defect or infirmity with which he has become affected since the payment of last preceding premium.

This policy may be renewed by mutual consent every year and in such event the renewal premium shall be paid to the Company on or before the date of expiry of the policy or of the subsequent renewal thereof. The Company shall not ,however, be bound to give notice that such renewal premium is due.

Possibility of revision of the premium rates:

The company, with prior approval of IRDA], may revise or modify the premium rates.

8. CANCELLATION

The Insured may cancel this Policy by giving 15days' written notice, and in such an event, the Company shall refund premium on short term rates for the unexpired Policy Period as per the rates detailed below.

The premium, on cancellation by insured, will be retained BY INSURER as follows:

Period On Risk

Rate Of Premium To Be Retained


Up to one month

1/4th of the annual rate

Up to three months

1/2 of the annual rate

Up to six months

3/4th of the annual rate

Exceeding six months

Full annual rate

i) Notwithstanding anything contained herein or otherwise, no refunds of premium shall be made in respect of Cancellation where, any claim has been admitted or has been lodged or any benefit has been availed by the Insured person under the Policy.

ii. The Company may cancel the Policy at any time on grounds of misrepresentation, non- disclosure of material facts, fraud by the Insured Person, by giving 15 days' written notice. There would be no refund of premium on cancellation on grounds of misrepresentation, non-disclosure of material facts or fraud.

9. NOMINATION
The insured person is required at the inception of the policy, to make a nomination of the eligible family pension recipient / recipients for the purpose of payment of claim under the policy in the event of Death of the policyholder. Any change in the nomination shall be communicated to the insurer in writing and such change shall be effective only when endorsement on the policy is made. In the event of death of the policyholder, Company will pay the nominee ( as named in the policy / pension scheme ) and in case there is no nominee ,to the legal representative of the policyholder whose discharge shall be treated as full and final discharge of its liability under the policy.

10. POLICY DISPUTES
Any dispute concerning the interpretation of the terms, conditions, limitations and/or exclusions contained herein is understood and agreed to by both the Insured and the Company to be subject to Indian Law.

11. ARBITRATION
If any dispute or difference shall arise to the quantum to be paid under the policy liability being otherwise admitted such difference shall independently be referred to the decision of a sole arbitrator to be appointed in writing by the parties to if they cannot agree upon a single arbitrator within 30 days of any party invoking arbitration the shall be referred to a panel of three arbitrators ,comprising of two arbitrators ,one to be appointed by each of the parties to the dispute / difference and the third arbitrator to be appointed by such two arbitrators and arbitration shall be conducted under and in accordance with the provisions of the Arbitration and Conciliation Act, 1996, as amended by Arbitration and Reconciliation ( Amendment ) Act, 2015 ( No.3 of 2016 ).

It is clearly agreed and understood that no difference or dispute shall be referable to arbitration as herein before provided, if the Company has disputed or not accepted liability under or in respect of this policy.

It is hereby expressly agreed and declared that it shall be a condition precedent to any right of action or suit upon this policy that award by such arbitrator / arbitrators of the amount of the loss or damage shall be first obtained.

It is also hereby further expressly agreed and declared that if the Company shall disclaim liability to the insured for any claim hereunder and such claim shall not within 12 calendar months from the date of such disclaimer have been made the subject matter of a suit in a Court of Law, the claim shall for all purposes be deemed to have been abandoned and shall not thereafter be recoverable hereunder.

12. FREE LOOK PERIOD
i) The free look period shall be applicable at the inception of the policy and

(1)The insured will be allowed a period of at least 15 days from the date of receipt of the policy to review the terms and conditions of the policy and to return the same if not acceptable.

(2) If the insured has not made any claim during the free look period, the insured shall be entitled to—(a) A refund of the premium paid less any expenses incurred by the insurer on medical examination of the insured persons and the stamp duty charges or; (b) where the risk has already commenced and the option of return of the policy is exercised by the policyholder, a deduction towards the proportionate risk premium for period on cover or; (c)Where only a part of the insurance coverage has commenced, such proportionate premium commensurate with the insurance coverage during such period; (d) In respect of unit linked policy, in addition to the above deductions, the insurer shall also be entitled to repurchase the unit at the price of the units as on the date of the return of the policy.

13. GRIEVANCE REDRESSAL
In case of any grievance the insured person may contact the company through

i.         Website: www.newindia.co.in ii. Toll free: 1800 209 1415 iii.        E-mail: As stated in the policy schedule iv.         Fax : As stated in the policy schedule

v. Courier: As stated in the policy schedule

Insured person may also approach the grievance cell at any of the company's branches with the details of grievance.

If Insured person is not satisfied with the redressal of grievance through one of the above methods, insured person may contact the grievance officer at New India Head Office.

For updated details of grievance officer, kindly refer the link at www.newindia.co.in

Insurance Ombudsman —The insured person may also approach the office of Insurance Ombudsman of the respective area/region for redressal of grievance. The contact details of the Insurance Ombudsman offices have been provided as Annexure-A. Insureds are advised to note the revised details of insurance ombudsman as and when amended as available in the website http://ecoi.co.in/ombudsman.html """
policy_19_text = """Office Protection Shield Insurance
Office Protection Shield Insurance

Product

Highlights





Cover for Building and Contents of an office against the perils like fire and allied perils, Burglary, Breakdown of Office equipments, Public Liability cover etc.



1.  Immediate notification of the loss to be given to Insurer.

2.  Immediate intimation to the Police and filing a FIR yin event of burglary / theft.

3.  Submission of claim form giving description of loss and supporting documents,

4.  Take immediate steps to save property as if it is uninsured.

Product FAQ



FAQ

 



1. Why Office Protection Shield?

It Provides an Umbrella Cover for your Office and enable you with a Worry Free Cover such as, Fire, Natures' Perils, Burglary, Machine Breakdown, Theft of Money in transit, Fidelity Protection, Additional expenses incurred for accommodation if the existing office is declared unfit for occupation, Loss of Baggage while on Journey, Bodily injury to any Third Party while using the office premise and Compensation to Workmen as Per Act.

2. Does it Carry any Co-Payment?

While Co-payment is not insisted, a small amount as Policy Deductible would be borne by you.

3. Whether I will get any Discounts?

As per MOU with 'CSI, your premium rates are heavily discounted. The rates shared with you are after discount and excludes GST.

4. Whether this policy carry Exclusions?

While we protect you and your office, a few causes like shortage due to errorr defaults due to design defect etc., are excluded.

5. Whether my employees can be insured under this policy?

Yes, especially for Money Insurance section, Fidelity Guarante, Baggage are a few sections that would be useful for your employees too as an Umbrella Protection.

6. Why should I buy multiple Insurance Policies when you call this as an Umbrella Cover?

http://bookstack.zubera.one/link/336#bkmrk-it-is-a-pertinent-qu
 
It is a pertinent Question. While some Insurance covers are compulsory like Motor Insurance, Unique policies like Mediclaim address the need basing on specific relation and purpose."""
policy_20_text = """Package Policy
Product

Highlights



PRODUCT HIGHLIGHTS
The policy is a combination of standard covers of various lines of businesses. This is a comprehensive policy that combines various products into one. It is beneficial for business enterprises.

1. Section l: Fire

Protection against losses caused by fire, including damage to property and contents.

2. Section Il: Engineering

Covers damage to engineering works, machinery, and equipment.

3. Section Ill: Burglary

Safeguards against theft and burglary-related damages.

4. Section IV: Money

Coverage for loss of money, including cash, cheques, and other forms of currency.

5. Section V: Plate Glass

Protection for damage to plate glass in your premises.

6. Section VI: Fidelity Guarantee

Covers losses due to dishonesty or fraud by employees.

7. Section Vll: Neon Sign
Package 

Policy

Insurance for damage to neon signs and their components.

8.    Section VI": Baggage

Protection for loss or damage to business-related baggage.

9.    Section 'X: Liability (restricted limits)

Covers liability for certain risks with specified limits.

10. Section X: Portable Equipment

Insurance for loss or damage to portable equipment used in business operations.

1 1. Section X': Workmen's Compensation

Covers compensation for work-related injuries to employees.

12. Section X' l: Accident Damage
Protection against accidental damage to property.

13. Section Xlll: Transit

Covers goods and equipment while in transit.

14. Section XV: Any Other Risk
Additional coverage for miscellaneous risks not covered in the above sections.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details.

CLAIMS PROCESS

1. Immediate notification of the loss to be given to Insurer.

2. Submission of claim form


 
 	

Provide a description of the loss along with supporting documents.

3. Furnish all such information and documentary evidence

Submit any additional information and documents as required by the Company."""
policy_21_text = """Overseas Mediclaim Insurance Policy (Employment & Studies)
Overseas Mediclaim Insurance Policy (Employment & Studies)

Highlights
Premium payable in Rupees and Claims settled abroad in foreign Currency. Policy available for frequent corporate travelers

Scope

Medical expenses incurred by the insured persons, outside India as a direct result of bodily injuries caused or sickness or disease contracted are covered. Following Plans are available:

OMP - BUSINESS AND HOLIDAY (B & H) PLANS


A- 1 (World        wide 50000            10000 1000            100         150         200000   100 excluding

USA/Canada)

A-               2(World  wide 250000          25000 1000            100         250         200000   100 excluding

USA/Canada)

B-l (World              wide 100000          25000 1000            100         150         200000   100 including

USA/Canada)

B-               2(World  wide 500000          25000 1000            100         250         200000   100 including

USA/Canada)

E-l CFT(World wide 100000 25000 1000            100         150         200000   100 including

USA/Canada)

E-2 CFT(World wide in 500000            25000 1000            100         250         200000   100 cluding USA/Canada)

CFT Cover is available for Executives of Coporate clients and Partners of registered firms annually subject to the duration of any one trip not exceeding 60 days.



C(Worldwide excluding USA/Canada)

*150000

10000

5000

100

D(Worldwide including USA/Canada)

*150000

10000

5000

100

D-1(Worldwide excluding USA/Canada)

*500000

10000

5000

100

*Contingency insurance for students US $750 for each month of completion of study during period of Insurance

Premium: Depends on Age-band, Trip-band and Country of visits.Coverage: Initially cover upto 180 days is provided under Business & Holiday Plan .Extension allowed on original policy for further period of 180 days subject to declaration of good health.

Eligibility
Age Limit: 6 months and above upto 70 years.

Policy is to be taken prior to departure from India. Medical Reports are required for:

A.         Trip is for period over 60 days and if

a.  insured person is over 60 yrs of age visiting USA/Canada

b. insured is over 70 yrs of age and visiting countries other than USA/Canada.

B. Proposal reveals that insured had suffered from/suffering from any illness/disease.

The Proposal Form should be accompanied with (1) ECG printout with report and (2) Fasting blood Sugar and Urine Sugar, Urine Strip Test Report or any other medical report required by the company etc. along with the attached questionnaire Il (B) to be completed and signed by the Doctor with minimum M. D. qualification conducting the test.

Major Exclusion

* All pre-existing disease/illnesses are not covered (known and unknown).

* Traveling against Medical advice or for Medical treatment including routine check-up.

* First USD 100 of all claims are to be borne by the traveller.

Please refer to Policy for further details.

Note: Policy details given are indicative, not exhaustive. Please contact your nearest NIA office for further details"""


policies = {
    "Shop Keeper Insurance": policy_1_text,
    "Rasta Apatti Kavach Policy": policy_2_text,
    "Plate Glass Insurance": policy_3_text,
    "Jewellers Block Insurance": policy_4_text,
    "House Holder Insurance": policy_5_text,
    "PROSPECTUS New India Pension Protect Personal Accident Policy" : policy_6_text,
    "Portable Equipment Insurance" : policy_7_text,
    "Bankers Indemnity Insurance" : policy_8_text,
    "Bhagyashree Insurance" : policy_9_text,
    "Burglary Insurance" : policy_10_text,
    "E Flight Coupon Insurance" : policy_11_text,
    "Event Cancellation Insurance" : policy_12_text,
    "Fidelity Guarantee Insurance" : policy_13_text,
    "JANATA PERSONAL ACCIDENT POLICY" : policy_14_text,
    "Mahila Udyam Bima" : policy_15_text,
    "Money Insurance" : policy_16_text,
    "Neon Sign Insurance" : policy_17_text,
    "NEW INDIA PENSION PROTECT PERSONAL ACCIDENT POLICY" : policy_18_text,
    "Office Protection Shield Insurance" : policy_19_text,
    "Package Policy" : policy_20_text,
    "Overseas Mediclaim Insurance Policy (Employment & Studies)" : policy_21_text
}

# ✅ Streamlit page config
st.set_page_config(page_title="Insurance Q&A Bot", layout="wide")
policy_names = list(policies.keys())
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Caching policy embeddings
@st.cache_resource
def get_policy_embeddings():
    return embedding_model.encode(list(policies.values()), convert_to_tensor=True)

policy_embeddings = get_policy_embeddings()


# ✅ Language options
indian_languages = {
    "English (Default)": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Odia": "or"
}

selected_language_name = st.selectbox("🌐 Select Output Language:", list(indian_languages.keys()))
selected_language_code = indian_languages[selected_language_name]

# ✅ Session states
if "memory" not in st.session_state:
    st.session_state.memory = deque(maxlen=10)
if "active_policies" not in st.session_state:
    st.session_state.active_policies = []
if "active_policies_reason" not in st.session_state:
    st.session_state.active_policies_reason = ""

# ✅ Input from user
st.title("📄 Insurance Q&A Bot with Memory")
question = st.text_input("💬 Ask your insurance-related question:")


# ✅ Select relevant policies (only on first query)
def select_top_policies_by_embedding(question, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_embedding, policy_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    return [policy_names[i] for i in top_results.indices]

# ✅ Clear memory button
if st.button("🧹 Start New Inquiry (Clear Memory)"):
    st.session_state.memory.clear()
    st.session_state.active_policies.clear()
    st.session_state.active_policies_reason = ""
    st.success("Memory and selected policies cleared.")

if st.button("🔁 Explore Other Policies"):
    if question.strip() == "":
        st.warning("Please enter a new query to explore other policies.")
    else:
        st.session_state.active_policies = select_top_policies_by_embedding(question)
        st.session_state.active_policies_reason = f"(Exploring: {question})"
        st.success("Now exploring different policies based on your latest query.")



# ✅ Ask Gemini with memory
def ask_ai_with_memory(question, selected_text):
    memory_context = ""
    for q, a in st.session_state.memory:
        if "i am" in q.lower() or "i'm" in q.lower():
            memory_context += f"User Info: {q}\n"
        else:
            memory_context += f"- Q: {q}\n  A: {a}\n"

    full_prompt = f"""
You are an intelligent insurance assistant helping users understand policy documents.

Use the following:
1. User's prior context and memory
2. Only the selected policies relevant to the current question

---
User Context:
{memory_context}

---
Relevant Policy Documents:
{selected_text}

---
Current Question:
{question}

Answer clearly based **only** on the relevant policies and previous conversation context. Avoid guessing or including unrelated policies.
"""
    response = model.generate_content(full_prompt)
    return response.text

# ✅ Translate response

def translate_answer(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source='auto', target=target_lang).translate(text)



with st.expander("🧠 View Conversation Memory"):
    for q, a in st.session_state.memory:
        st.markdown(f"**Q:** {q}\n\n**A:** {a}")

# ✅ Main Q&A flow
if question:
    with st.spinner("🔍 Identifying relevant policies..."):
        if not st.session_state.active_policies:
            top_policies = select_top_policies_by_embedding(question)
            st.session_state.active_policies = top_policies
            st.session_state.active_policies_reason = question
        else:
            top_policies = st.session_state.active_policies

    selected_text = "\n\n".join(policies[name] for name in top_policies)
    st.info(f"📘 Using these policies (based on: '{st.session_state.active_policies_reason}'):\n- " + "\n- ".join(top_policies))

    with st.spinner("🧠 Generating answer..."):
        answer = ask_ai_with_memory(question, selected_text)
        translated_answer = translate_answer(answer, selected_language_code)
        st.session_state.memory.append((question, translated_answer))

    st.subheader(f"🗣️ Answer in {selected_language_name}:")
    st.write(translated_answer)

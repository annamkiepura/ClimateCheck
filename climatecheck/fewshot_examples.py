"""Shared few-shot examples used by RankGPT ranking and LLM classification."""

# ── RankGPT permutation ranking examples ────────────────────────────────────
# Each item has a claim and ranked passages (evidence first, NEI last).

RANKGPT_FEWSHOT_EXAMPLES = [
    {
        "claim": "Burning natural gas does not release greenhouse gases.",
        "items": [
            {
                "id": 94021,
                "text": "Continuous stack monitoring at three combined-cycle natural-gas power stations (total output = 2.7 GW) was conducted over a full seasonal cycle. Infra-red gas analysers measured carbon-dioxide concentrations every minute, while fuel flow-meters provided simultaneous methane burn rates. After correcting for oxygen dilution, the plants emitted a mean value of 366 \u00b1 12 g CO\u2082 kWh\u207b\u00b9 of electricity delivered, in line with Intergovernmental Panel on Climate Change default factors. The study confirms that even 'high-efficiency' natural-gas combustion **does produce significant greenhouse-gas emissions**, directly contradicting assertions that it is a zero-carbon energy source.",
                "label": "refutes",
            },
            {
                "id": 94022,
                "text": "A cradle-to-grave life-cycle assessment examined extraction, processing, transmission and end-use combustion of natural gas in the Permian, Marcellus and North Sea basins. Field measurements show that fugitive methane leakage ranges from 1.6 % to 3.8 % of production, while routine venting during liquids unloading adds a further 0.4 %. When converted to carbon-dioxide equivalents over a 100-year horizon, upstream losses increase the effective climate footprint of gas-fired electricity by **15\u201325 %** relative to stack emissions alone. These findings **refute claims** that natural gas can be considered climate-neutral even before combustion takes place.",
                "label": "refutes",
            },
            {
                "id": 94023,
                "text": "A two-week pilot trial cofiring purified landfill gas (60 % CH\u2084, 40 % CO\u2082) in a 30 kW micro-turbine recorded outlet CO\u2082 levels of 1.7 \u00b1 0.4 %\u2014statistically indistinguishable from baseline calibration drift. However, the short duration, small thermal input and atypical fuel composition prevent robust extrapolation to utility-scale systems. The authors therefore conclude that the experiment provides **no definitive evidence** either for or against greenhouse-gas emissions from natural-gas combustion and emphasise the need for longer, higher-power trials.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Ocean acidification has no impact on shell-forming marine organisms.",
        "items": [
            {
                "id": 93011,
                "text": "A meta-analysis of 228 experiments shows reduced calcification rates (\u221222 %) in molluscs exposed to elevated CO\u2082 (pH 7.8) compared with present-day conditions, demonstrating that ocean acidification negatively affects shell formation.",
                "label": "refutes",
            },
            {
                "id": 93012,
                "text": "Larval oysters reared under acidified seawater (1 000 \u00b5atm CO\u2082) developed **significantly lower** shell mass and higher dissolution rates than controls, indicating that decreased pH harms calcifying organisms.",
                "label": "refutes",
            },
            {
                "id": 93013,
                "text": "Long-term exposure (12 months) of juvenile crabs to moderately elevated CO\u2082 showed no statistically significant change in carapace thickness, suggesting species-specific responses to acidification.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Interesting read: a new article claims there's no evidence that human emissions of CO2 are driving global warming.",
        "items": [
            {
                "id": 192612,
                "text": "This article builds on the premise that human consumption of goods, food and transport are the ultimate drivers of climate change. However, the nature of the climate change problem (well described as a tragedy of the commons) makes it difficult for individuals to recognise their personal duty to implement behavioural changes to reduce greenhouse gas emissions. Consequently, this article aims to analyse the climate change issue from a human-scale perspective, in which each of us has a clearly defined personal quota of CO2 emissions that limits our activity and there is a finite time during which CO2 emissions must be eliminated to achieve the well below 2\u00b0C warming limit set by the Paris Agreement of 2015 (COP21). Thus, this work's primary contribution is to connect an equal per capita fairness approach to a global carbon budget, linking personal levels with planetary levels. Here, we show that a personal quota of 5.0 tons of CO2 yr-1 p-1 is a representative value for both past and future emissions; for this level of a constant per-capita emissions and without considering any mitigation, the global accumulated emissions compatible with the well below 2\u00b0C and 2\u00b0C targets will be exhausted by 2030 and 2050, respectively. These are references years that provide an order of magnitude of the time that is left to reverse the global warming trend. More realistic scenarios that consider a smooth transition toward a zero-emission world show that the global accumulated emissions compatible with the well below 2\u00b0C and 2\u00b0C targets will be exhausted by 2040 and 2080, respectively. Implications of this paper include a return to personal responsibility following equity principles among individuals, and a definition of boundaries to the personal emissions of CO2.",
                "label": "refutes",
            },
            {
                "id": 36931,
                "text": "It is generally accepted in the scientific community that carbon dioxide (CO2) emissions, which lead to global warming, arise from using fossil fuels, namely coal, oil and gas, as energy sources. Consequently, alleviating the effects of global warming and climate change necessitates substantial reductions in the use of fossil fuel energy. This paper uses a financial market-based approach to investigate whether positive stock returns cause changes in CO2 emissions, or vice-versa, based on the Granger causality test to determine cause and effect, or leader and follower. If Granger causality can be determined in any direction, this will enable a clear directional statement regarding temporal predictability between stock returns and CO2 emissions. The empirical data include annual CO2 emissions from fuel combustion of the three main fossil energy sources, namely coal, oil and gas, based on 18 countries with sophisticated financial markets that are in the Morgan Stanley Capital International (MSCI) World Index from 1971 to 2017. The empirical results show clearly that all the statistically significant causality findings are unidirectional from the stock market returns to CO2 emissions from coal, oil and gas, but not the reverse. More importantly, the regression results suggest that when stock returns rise by 1%, CO2 emissions from coal combustion decrease by 9% among the countries that are included in MSCI World Index. Furthermore, when stock returns rise 1%, CO2 emissions from oil combustion increase by 2%, but stock returns have no significant effect on CO2 emissions from gas combustion.",
                "label": "refutes",
            },
            {
                "id": 189578,
                "text": "It is now widely accepted that carbon emission from human activities is an important driving force in global warming, and global change has a deep impact on sustainable development of human society. To meet the challenges of global change, the international community has reached a consensus that developed countries take strict actions in emission reduction, whereas developing countries take spontaneous efforts in reducing emissions under the guiding principle of common but differentiated responsibilities, with an agreed goal to restrict global surface temperature increase due to human activities to within 2\u00b0C of pre-industrial levels. However, there is no clear pathway to reach this goal. A number of related questions must be addressed on principles to be followed, research emphasis and policy measures. Here we argue that response policies to address global change issues must be based on balanced development at regional and international levels, and on advancements in science and technology. This requires consideration of harmony not only between humans and nature but also within human societies, to properly deal with the relationship between global change and sustainable development. We must make equal efforts toward carbon emission reduction and carbon sequestration, and toward mitigation and adaptation. There should be more research support to reduce uncertainties in our understanding of global change. Addressing the challenges of global change creates great opportunities for the development of human society. This will facilitate transformation of energy use structure, improve and restore ecological functioning of the earth environment, transform production modes and ways of living in human society, and promote harmonic and balanced development at regional and international levels.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Planting trees will not prevent climate change.",
        "items": [
            {
                "id": 30576,
                "text": "Trees absorb carbon. Planting more trees will absorb more carbon from the atmosphere, and soak up the man\u2010made emissions that are causing climate change. It is a simple, easy and attractive solution that would allow us to continue our high\u2010emission business as usual and still stave off global warming. Brendan Mackey examines whether or not it would work.",
                "label": "refutes",
            },
            {
                "id": 13483,
                "text": "Tree planting is increasingly being proposed as a strategy to combat climate change through carbon (C) sequestration in tree biomass. However, total ecosystem C storage that includes soil organic C (SOC) must be considered to determine whether planting trees for climate change mitigation results in increased C storage. We show that planting two native tree species (Betula pubescens and Pinus sylvestris), of widespread Eurasian distribution, onto heather (Calluna vulgaris) moorland with podzolic and peaty podzolic soils in Scotland, did not lead to an increase in net ecosystem C stock 12 or 39 years after planting. Plots with trees had greater soil respiration and lower SOC in organic soil horizons than heather control plots. The decline in SOC cancelled out the increment in C stocks in tree biomass on decadal timescales. At all four experimental sites sampled, there was no net gain in ecosystem C stocks 12\u201339 years after afforestation\u2014indeed we found a net ecosystem C loss in one of four sites with deciduous B. pubescens stands; no net gain in ecosystem C at three sites planted with B. pubescens; and no net gain at additional stands of P. sylvestris. We hypothesize that altered mycorrhizal communities and autotrophic C inputs have led to positive 'priming' of soil organic matter, resulting in SOC loss, constraining the benefits of tree planting for ecosystem C sequestration. The results are of direct relevance to current policies, which promote tree planting on the assumption that this will increase net ecosystem C storage and contribute to climate change mitigation. Ecosystem\u2010level biogeochemistry and C fluxes must be better quantified and understood before we can be assured that large\u2010scale tree planting in regions with considerable pre\u2010existing SOC stocks will have the intended policy and climate change mitigation outcomes.",
                "label": "supports",
            },
            {
                "id": 59417,
                "text": "The prevailing nature\u2010based solution to tackle climate change is tree planting. However, there is growing evidence that it has serious contraindications in many regions. The main shortcoming of global tree planting is its awareness disparity to alternative ecosystem types, mainly grasslands. Grasslands, where they constitute the natural vegetation, may support higher biodiversity and a safer, soil\u2010locked carbon stock than plantations and other forests. We suggest replacing \u201ctree planting\u201d by \u201crestore native vegetation.\u201d This improved action terminology reduces the risks of inappropriate afforestation and, by diversifying target ecosystem types, does not reduce but increases potential land area for nature\u2010based climate mitigation.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Looks like climate models might be overestimating the warming trend. #ClimateAction #ClimateData'",
        "items": [
            {
                "id": 364646,
                "text": "Most present-generation climate models simulate an increase in global-mean surface temperature (GMST) since 1998, whereas observations suggest a warming hiatus. It is unclear to what extent this mismatch is caused by incorrect model forcing, by incorrect model response to forcing or by random factors. Here we analyse simulations and observations of GMST from 1900 to 2012, and show that the distribution of simulated 15-year trends shows no systematic bias against the observations. Using a multiple regression approach that is physically motivated by surface energy balance, we isolate the impact of radiative forcing, climate feedback and ocean heat uptake on GMST\u2014with the regression residual interpreted as internal variability\u2014and assess all possible 15- and 62-year trends. The differences between simulated and observed trends are dominated by random internal variability over the shorter timescale and by variations in the radiative forcings used to drive models over the longer timescale. For either trend length, spread in simulated climate feedback leaves no traceable imprint on GMST trends or, consequently, on the difference between simulations and observations. The claim that climate models systematically overestimate the response to radiative forcing from increasing greenhouse gas concentrations therefore seems to be unfounded.",
                "label": "refutes",
            },
            {
                "id": 52140,
                "text": "Multi\u2010model climate experiments carried out as part of different phases of the Coupled Model Intercomparison Project (CMIP) are crucial to evaluate past and future climate change. The reliability of models' simulations is often gauged by their ability to reproduce the historical climate across many time scales. This study compares the global mean surface air temperature from 29 CMIP6 models with observations from three datasets. We examine (1) warming and cooling rates in five subperiods from 1880 to 2014, (2) autocorrelation and long\u2010term persistence, (3) models' performance based on probabilistic and entropy metrics, and (4) the distributional shape of temperature. All models simulate the observed long\u2010term warming trend from 1880 to 2014. The late twentieth century warming (1975\u20132014) and the hiatus (1942\u20131975) are replicated by most models. The post\u20101998 warming is overestimated in 90% of the simulations. Only six out of 29 models reproduce the observed long\u2010term persistence. All models show differences in distributional shape when compared with observations. Varying performance across metrics reveals the challenge to determine the \u201cbest\u201d model. Thus, we argue that models should be selected, based on case\u2010specific metrics, depending on the intended use. Metrics proposed here facilitate a comprehensive assessment for various applications.",
                "label": "supports",
            },
            {
                "id": 377418,
                "text": "Air pressure at sea level during winter has decreased over the Arctic and increased in the Northern Hemisphere subtropics in recent decades, a change that has been associated with 50% of the Eurasian winter warming observed over the past 30 years, with 60% of the rainfall increase in Scotland and with 60% of the rainfall decrease in Spain. This trend is inconsistent with the simulated response to greenhouse-gas and sulphate-aerosol changes, but it has been proposed that other climate influences--such as ozone depletion--could account for the discrepancy. Here I compare observed Northern Hemisphere sea-level pressure trends with those simulated in response to all the major human and natural climate influences in nine state-of-the-art coupled climate models over the past 50 years. I find that these models all underestimate the circulation trend. This inconsistency suggests that we cannot yet simulate changes in this important property of the climate system or accurately predict regional climate changes.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "'natural gas' is considered cleaner than coal and oil",
        "items": [
            {
                "id": 7896,
                "text": "In April 2011, we published the first peer\u2010reviewed analysis of the greenhouse gas footprint (GHG) of shale gas, concluding that the climate impact of shale gas may be worse than that of other fossil fuels such as coal and oil because of methane emissions. We noted the poor quality of publicly available data to support our analysis and called for further research. Our paper spurred a large increase in research and analysis, including several new studies that have better measured methane emissions from natural gas systems. Here, I review this new research in the context of our 2011 paper and the fifth assessment from the Intergovernmental Panel on Climate Change released in 2013. The best data available now indicate that our estimates of methane emission from both shale gas and conventional natural gas were relatively robust. Using these new, best available data and a 20\u2010year time period for comparing the warming potential of methane to carbon dioxide, the conclusion stands that both shale gas and conventional natural gas have a larger GHG than do coal or oil, for any possible use of natural gas and particularly for the primary uses of residential and commercial heating. The 20\u2010year time period is appropriate because of the urgent need to reduce methane emissions over the coming 15\u201335 years.",
                "label": "refutes",
            },
            {
                "id": 186007,
                "text": 'A well-known theorem by Herfindahl states that the low-cost nonrenewable resource must be exploited first. Consider resources that are differentiated only by their pollution content. For instance, both coal and natural gas are used to generate electricity, yet coal is more polluting. We show that the ordering of extraction need not be driven by whether a resource is clean or dirty. Coal may be used first, followed by natural gas, and again by coal. Such \u201cvacillation\u201d does not occur under cost heterogeneity. A perverse policy implication is that regulating pollution may accelerate use of the polluting resource.(JEL Q32, Q38, Q53, Q58)',
                "label": "supports",
            },
            {
                "id": 36069,
                "text": 'Shale gas proponents argue this unconventional fossil fuel offers a \u201cbridge\u201d towards a cleaner energy system by offsetting higher-carbon fuels such as coal. The technical feasibility of reconciling shale gas development with climate action remains contested. However, we here argue that governance challenges are both more pressing and more profound. Reconciling shale gas and climate action requires institutions capable of responding effectively to uncertainty; intervening to mandate emissions reductions and internalize costs to industry; and managing the energy system strategically towards a lower carbon future. Such policy measures prove challenging, particularly in jurisdictions that stand to benefit economically from unconventional fuels. We illustrate this dilemma through a case study of shale gas development in British Columbia, Canada, a global leader on climate policy that is nonetheless struggling to manage gas development for mitigation. The BC case is indicative of the constraints jurisdictions face both to reconcile gas development and climate action, and to manage the industry adequately to achieve social licence and minimize resistance. More broadly, the case attests to the magnitude of change required to transform our energy systems to mitigate climate change.',
                "label": "nei",
            },
        ],
    },
]

# ── LLM classification few-shot examples (hybrid zero-shot variant) ─────────
# Same structure but used for per-abstract classification prompts.

LLM_CLASSIFICATION_FEWSHOT_EXAMPLES = RANKGPT_FEWSHOT_EXAMPLES


# ── LLM classification few-shot examples (hybrid few-shot variant) ──────────
# A different set used for the few-shot classification approach.

LLM_FEWSHOT_VARIANT_EXAMPLES = [
    {
        "claim": "Claiming that climate change is a data manipulation is dangerous and misleading.",
        "items": [
            {
                "id": 72260,
                "text": "Misinformation can have significant societal consequences. For example, misinformation about climate change has confused the public and stalled support for mitigation policies. When people lack the expertise and skill to evaluate the science behind a claim, they typically rely on heuristics such as substituting judgment about something complex (i.e. climate science) with judgment about something simple (i.e. the character of people who speak about climate science) and are therefore vulnerable to misleading information. Inoculation theory offers one approach to effectively neutralize the influence of misinformation. Typically, inoculations convey resistance by providing people with information that counters misinformation. In contrast, we propose inoculating against misinformation by explaining the fallacious reasoning within misleading denialist claims. We offer a strategy based on critical thinking methods to analyse and detect poor reasoning within denialist claims. This strategy includes detailing argument structure, determining the truth of the premises, and checking for validity, hidden premises, or ambiguous language. Focusing on argument structure also facilitates the identification of reasoning fallacies by locating them in the reasoning process. Because this reason-based form of inoculation is based on general critical thinking methods, it offers the distinct advantage of being accessible to those who lack expertise in climate science. We applied this approach to 42 common denialist claims and find that they all demonstrate fallacious reasoning and fail to refute the scientific consensus regarding anthropogenic global warming. This comprehensive deconstruction and refutation of the most common denialist claims about climate change is designed to act as a resource for communicators and educators who teach climate science and/or critical thinking.",
                "label": "supports",
            },
            {
                "id": 74507,
                "text": 'Science is based on a shared respect for the scientific method\u2014the principle that, by gathering and analyzing data and information, scientists and others can draw conclusions that are robust and generalizable across cultures and ideologies. Scientists furthermore assume that disagreements can be resolved by more facts. So when people object to the reality of climate change with science-y sounding arguments\u2014\u201cthe data is wrong,\u201d or \u201cit\u2019s just a natural cycle,\u201d or even, \u201cwe need to study it longer\u201d\u2014the natural response of scientists is simple and direct: People need more data. But this approach often doesn\u2019t work and can even backfire. Why? Because when it comes to climate change, science-y sounding objections are a mere smokescreen to hide the real reasons, which have much more to do with identity and ideology than data and facts.',
                "label": "refutes",
            },
            {
                "id": 827,
                "text": "Greenhouse gas (GHG) emissions are exter nalities and represent the biggest market failure the world has seen. We all produce emissions, people around the world are already suffering from past emissions, and current emissions will have potentially catastrophic impacts in the future. Thus, these emissions are not ordinary, localized externalities. Risk on a global scale is at the core of the issue. These basic features of the problem must shape the economic analy sis we bring to bear; failure to do this will, and has, produced approaches to policy that are pro foundly misleading and indeed dangerous. The purpose of this lecture is to set out what I think is an appropriate way to examine the economics of climate change, given the unique scientific and economic challenges posed, and to suggest implications for emissions targets, policy instruments, and global action. The sub ject is complex and very wide-ranging. It is a subject of vital importance but one in which the economics is fairly young. A central challenge is to provide the economic tools necessary as",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "New research suggests CO2 levels might not be the main driver of rising temperatures.",
        "items": [
            {
                "id": 280794,
                "text": "Atmospheric levels of CO2 are commonly assumed to be a main driver of global climate. Independent empirical evidence suggests that the galactic cosmic ray flux (CRF) is linked to climate variability. Both drivers are presently discussed in the context of daily to millennial variations, although they should also operate over geological time scales. Here we analyze the reconstructed seawater paleotemperature record for the Phanerozoic (past 545 m.y.), and compare it with the variable CRF reaching Earth and with the reconstructed partial pressure of atmospheric CO2 (pCO2). We find that at least 66% of the variance in the paleotemperature trend could be attributed to CRF variations likely due to solar system passages through the spiral arms of the galaxy. Assuming that the entire residual variance in temperature is due solely to the CO2 greenhouse effect, we propose a tentative upper limit to the long-term \u201cequilibrium\u201d warming effect of CO2, one which is potentially lower than that based on general circulation models.",
                "label": "supports",
            },
            {
                "id": 190177,
                "text": "It is well established that carbon dioxide (CO2) is the most prominent agent of climate change. The level of CO2 in the atmosphere has been increasing persistently over the last few decades due to rising dependence on fossil fuels for energy production. India is facing a potential energy crisis. India has large coal reserves and coal is currently the linchpin of the Indian power sector, making Indian coal-derived emissions a focus of global attention. Further, India's journey from a challenging energy security situation to the 'Make in India' initiative is expected to drive energy needs exponentially. Thus, in the context of a rapidly changing climate, it has become imperative to quantify the emissions of greenhouse gases (GHGs) from emerging coal-based energy plants in India. The present work attempts not only to do this, with the intention of highlighting India's commitment to reducing CO2 emissions, but also to redefine India's future emissions. We draw attention to India's attempt to transform the coal technology used in coal-based thermal power plants. We have tried to adopt a holistic approach to quantify the past (2010), present (2015) and future (2025) emission trends for important GHGs like CO2 and other critical air pollutants from rapidly penetrating low-emission advanced coal technology. Our estimation shows that CO2 emissions will increase from 1065 Tg yr\u22121 (2015) to 2634 Tg yr\u22121 (2025), which is approximately 147% of the current value. This rapid increase is largely attributed to rising energy demand due to industrial development, followed by demand from the domestic and agricultural sectors. The present trend of CO2 emissions is sure to propel India to become world's second largest emitter of GHGs in 2025, dislodging the United States. We have also estimated the emission of other pollutants like NOx, SO2, black carbon, organic carbon, particulate matter (PM2.5, PM10), volatile organic compounds and CO. Our findings seem to suggest that India will able to cut CO2 emission from the traditionally dominant thermal power sector by at least 19% in 2025. Present attempts at emission reduction, along with the government's massive initiatives towards building renewable energy infrastructure, could be well aligned to India's Intended Nationally Determined Contribution submission to COP21 of the United Nations Framework Convention on Climate Change. With such a rapid expansion of energy production it can be assumed that cost-effective and uninterrupted power (i.e. 24/7) can be provided to all citizens of the country well before 2025.",
                "label": "refutes",
            },
            {
                "id": 59516,
                "text": 'Abstract. Climate change is projected to increase the imbalance between the supply (precipitation) and atmospheric demand for water (i.e., increased potential evapotranspiration), stressing plants in water-limited environments. Plants may be able to offset increasing aridity because rising CO2 increases water use efficiency. CO2 fertilization has also been cited as one of the drivers of the widespread \u201cgreening\u201d phenomenon. However, attributing the size of this CO2 fertilization effect is complicated, due in part to a lack of long-term vegetation monitoring and interannual- to decadal-scale climate variability. In this study we asked the question of how much CO2 has contributed towards greening. We focused our analysis on a broad aridity gradient spanning eastern Australia\u2019s woody ecosystems. Next we analyzed 38 years of satellite remote sensing estimates of vegetation greenness (normalized difference vegetation index, NDVI) to examine the role of CO2 in ameliorating climate change impacts. Multiple statistical techniques were applied to separate the CO2-attributable effects on greening from the changes in water supply and atmospheric aridity. Widespread vegetation greening occurred despite a warming climate, increases in vapor pressure deficit, and repeated record-breaking droughts and heat waves. Between 1982\u20132019 we found that NDVI increased (median 11.3 %) across 90.5 % of the woody regions. After masking disturbance effects (e.g., fire), we statistically estimated an 11.7 % increase in NDVI attributable to CO2, broadly consistent with a hypothesized theoretical expectation of an 8.6 % increase in water use efficiency due to rising CO2. In contrast to reports of a weakening CO2 fertilization effect, we found no consistent temporal change in the CO2 effect. We conclude rising CO2 has mitigated the effects of increasing aridity, repeated record-breaking droughts, and record-breaking heat waves in eastern Australia. However, we were unable to determine whether trees or grasses were the primary beneficiary of the CO2-induced change in water use efficiency, which has implications for projecting future ecosystem resilience. A more complete understanding of how CO2-induced changes in water use efficiency affect trees and non-tree vegetation is needed.',
                "label": "nei",
            },
        ],
    },
    {
        "claim": "New study suggests global warming might be less severe than previously predicted by models. #ClimateChange #Science",
        "items": [
            {
                "id": 379964,
                "text": "An irreducibly simple climate-sensitivity model is designed to empower even non-specialists to research the question how much global warming we may cause. In 1990, the First Assessment Report of the Intergovernmental Panel on Climate Change (IPCC) expressed 'substantial confidence' that near-term global warming would occur twice as fast as subsequent observation. Given rising CO2 concentration, few models predicted no warming since 2001. Between the pre-final and published drafts of the Fifth Assessment Report, IPCC cut its near-term warming projection substantially, substituting 'expert assessment' for models' near-term predictions. Yet its long-range predictions remain unaltered. The model indicates that IPCC's reduction of the feedback sum from 1.9 to 1.5 W m\u22122 K\u22121 mandates a reduction from 3.2 to 2.2 K in its central climate-sensitivity estimate; that, since feedbacks are likely to be net-negative, a better estimate is 1.0 K; that there is no unrealized global warming in the pipeline; that global warming this century will be <1 K; and that combustion of all recoverable fossil fuels will cause <2.2 K global warming to equilibrium. Resolving the discrepancies between the methodology adopted by IPCC in its Fourth and Fifth Assessment Reports that are highlighted in the present paper is vital. Once those discrepancies are taken into account, the impact of anthropogenic global warming over the next century, and even as far as equilibrium many millennia hence, may be no more than one-third to one-half of IPCC's current projections.",
                "label": "supports",
            },
            {
                "id": 128897,
                "text": "Studies indicate that, historically, terrestrial ecosystems of the northern high-latitude region may have been responsible for up to 60% of the global net land-based sink for atmospheric CO2. However, these regions have recently experienced remarkable modification of the major driving forces of the carbon cycle, including surface air temperature warming that is significantly greater than the global average and associated increases in the frequency and severity of disturbances. Whether Arctic tundra and boreal forest ecosystems will continue to sequester atmospheric CO2 in the face of these dramatic changes is unknown. Here we show the results of model simulations that estimate a 41 Tg C yr\u22121 sink in the boreal land regions from 1997 to 2006, which represents a 73% reduction in the strength of the sink estimated for previous decades in the late 20th century. Our results suggest that CO2 uptake by the region in previous decades may not be as strong as previously estimated. The recent decline in sink strength is the combined result of (1) weakening sinks due to warming-induced increases in soil organic matter decomposition and (2) strengthening sources from pyrogenic CO2 emissions as a result of the substantial area of boreal forest burned in wildfires across the region in recent years. Such changes create positive feedbacks to the climate system that accelerate global warming, putting further pressure on emission reductions to achieve atmospheric stabilization targets.",
                "label": "refutes",
            },
            {
                "id": 354118,
                "text": "The IPCC has reasserted the strong influence of anthropogenic CO2 contributions on global climate change and highlighted the polar-regions as highly vulnerable. With these predictions the cold adapted fauna endemic to the Southern Ocean, which is dominated by fishes of the sub-order Notothenioidei, will face considerable challenges in the near future. Recent physiological studies have demonstrated that the synergistic stressors of elevated temperature and ocean acidification have a considerable, although variable, impact on notothenioid fishes. The present study explored the transcriptomic response of Pagothenia borchgrevinki to increased temperatures and pCO2 after 7, 28 and 56days of acclimation. We compared this response to short term studies assessing heat stress alone and foretell the potential impacts of these stressors on P. borchgrevinki's ability to survive a changing Southern Ocean.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Planting trees will not prevent climate change.",
        "items": [
            {
                "id": 30576,
                "text": "Trees absorb carbon. Planting more trees will absorb more carbon from the atmosphere, and soak up the man\u2010made emissions that are causing climate change. It is a simple, easy and attractive solution that would allow us to continue our high\u2010emission business as usual and still stave off global warming. Brendan Mackey examines whether or not it would work.",
                "label": "refutes",
            },
            {
                "id": 13483,
                "text": "Tree planting is increasingly being proposed as a strategy to combat climate change through carbon (C) sequestration in tree biomass. However, total ecosystem C storage that includes soil organic C (SOC) must be considered to determine whether planting trees for climate change mitigation results in increased C storage. We show that planting two native tree species (Betula pubescens and Pinus sylvestris), of widespread Eurasian distribution, onto heather (Calluna vulgaris) moorland with podzolic and peaty podzolic soils in Scotland, did not lead to an increase in net ecosystem C stock 12 or 39 years after planting.",
                "label": "supports",
            },
            {
                "id": 59417,
                "text": "The prevailing nature\u2010based solution to tackle climate change is tree planting. However, there is growing evidence that it has serious contraindications in many regions. The main shortcoming of global tree planting is its awareness disparity to alternative ecosystem types, mainly grasslands.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "Looks like climate models might be overestimating the warming trend. #ClimateAction #ClimateData'",
        "items": [
            {
                "id": 364646,
                "text": "Most present-generation climate models simulate an increase in global-mean surface temperature (GMST) since 1998, whereas observations suggest a warming hiatus. It is unclear to what extent this mismatch is caused by incorrect model forcing, by incorrect model response to forcing or by random factors. Here we analyse simulations and observations of GMST from 1900 to 2012, and show that the distribution of simulated 15-year trends shows no systematic bias against the observations.",
                "label": "refutes",
            },
            {
                "id": 52140,
                "text": "Multi\u2010model climate experiments carried out as part of different phases of the Coupled Model Intercomparison Project (CMIP) are crucial to evaluate past and future climate change. The post\u20101998 warming is overestimated in 90% of the simulations.",
                "label": "supports",
            },
            {
                "id": 377418,
                "text": "Air pressure at sea level during winter has decreased over the Arctic and increased in the Northern Hemisphere subtropics in recent decades. This inconsistency suggests that we cannot yet simulate changes in this important property of the climate system or accurately predict regional climate changes.",
                "label": "nei",
            },
        ],
    },
    {
        "claim": "'natural gas' is considered cleaner than coal and oil",
        "items": [
            {
                "id": 7896,
                "text": "In April 2011, we published the first peer\u2010reviewed analysis of the greenhouse gas footprint (GHG) of shale gas, concluding that the climate impact of shale gas may be worse than that of other fossil fuels such as coal and oil because of methane emissions.",
                "label": "refutes",
            },
            {
                "id": 186007,
                "text": 'A well-known theorem by Herfindahl states that the low-cost nonrenewable resource must be exploited first. Consider resources that are differentiated only by their pollution content. For instance, both coal and natural gas are used to generate electricity, yet coal is more polluting. We show that the ordering of extraction need not be driven by whether a resource is clean or dirty.(JEL Q32, Q38, Q53, Q58)',
                "label": "supports",
            },
            {
                "id": 36069,
                "text": 'Shale gas proponents argue this unconventional fossil fuel offers a \u201cbridge\u201d towards a cleaner energy system by offsetting higher-carbon fuels such as coal. The technical feasibility of reconciling shale gas development with climate action remains contested.',
                "label": "nei",
            },
        ],
    },
]

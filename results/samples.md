# Qualitative samples (Section 4.6)

Samples 1–3 are the posts where the reward model prefers PPO/DPO most over SFT, 4–6 where it prefers SFT most, and
7–10 are random. Clearly better than SFT: 1, 2, 3 (PPO), 8 (DPO), 9 (PPO). Clearly worse: 4 (PPO and DPO), 5 (PPO),
6 (DPO). Across all ten, PPO and DPO are more specific and closer to the post's wording than SFT; their failures are
repetition, drifting into minor details and truncation at 48 tokens.

## Sample 1 (eval post 206; picked because: reward model prefers PPO/DPO most over SFT)

```
SUBREDDIT: r/offmychest
TITLE: So I met an awesome girl last week and go study abroad in two weeks from today...
POST: Before I start this, here's some background as to who I am: a third-year computer science major at an engineering school who has never had a real girlfriend and only one kinda dated one. No relationships during highschool and none during my sophmore year of college.

The story is that I've been an intern since the beginning of June in NYC and she started as a new hire at the beginning of July. For all of July pretty much, we were pretty good friends and out of the blue, (to me) she invited me back to her apartment last Thursday after a going away party for a coworker. Now I really don't want to leave this fucking city but I also know I can't justify or sustain staying. My only plan right now is to keep in contact and remain friends until I can come back and visit and see where it's gone from there. Sorry, I just had to rant at someone about this and all my friends in the city work at my office.
TL;DR:
```

- **Human reference:** Intern meets girl. They become friends. Girl uses hookup on intern. Due to lack of time left together, it's highly effective! Intern is confused, annoyed, and maybe a little bit heartbroken.
- **SFT** (r_phi -1.87, 35 tokens): I'm a third-year intern at a university, I'm a girl, and I'm in a relationship. I'm not sure if I should stay or go.
- **PPO** (r_phi +4.60, 48 tokens): Met a girl last week and go study abroad in two weeks from today. Stay in contact and stay friends until I can come back and visit and see where it's gone from there. Don't want to leave this city but I also know
- **DPO** (r_phi +0.52, 31 tokens): Girlfriend invited me back to her apartment after a going away party for a coworker, I'm really not sure if I should stay or go.

**Why:** **PPO is clearly better than SFT.** SFT invents facts ("I'm a girl, and I'm in a relationship"); PPO gets the real situation — met a girl, leaving to study abroad, wants to stay in touch — but it is cut off at 48 tokens and copies the title. DPO is also wrong ("Girlfriend invited me…").

## Sample 2 (eval post 264; picked because: reward model prefers PPO/DPO most over SFT)

```
SUBREDDIT: r/AskReddit
TITLE: What would you dare or challenge me to do for a year, starting today, to make my life more fulfilling and more purposeful?
POST: When I was a kid, I've always wanted to be a rebel with an extraordinary life who made an impact. Unfortunately, I ended up in a vanilla state of things: I'm F/21, living in US, working a full time job, and being part-time in school. No SO at this point. I am not satisfied with where I am. I've always wanted more and I feel like now is the time for more.

My job pays me well so I have reasonable financial resources. I create my own schedule at work so my time is fairly flexible. Give me ideas - anything I can do on either a daily, weekly, or monthly basis (but something that doesn't interfere too much with my daily life; e.g. I can't move to a different state or country).
TL;DR:
```

- **Human reference:** I have a great life, but it's kind of boring... I want to be more impactful and have a more fulfilling life. What do you challenge me to do for this following year?
- **SFT** (r_phi -1.94, 35 tokens): I want to be a rebel, but I don't know what to do. I'm a 21 year old, and I'm not sure what to do.
- **PPO** (r_phi +2.83, 48 tokens): I want to make my life more fulfilling and more purposeful. I've always wanted more and I feel like now is the time for more. What would you dare or challenge me to do for a year, starting today, to make my life
- **DPO** (r_phi +2.85, 45 tokens): I want to be a rebel with an extraordinary life who makes an impact on my life. What would you dare or challenge me to do for a year, starting today, to make my life more fulfilling and more purposeful?

**Why:** **PPO and DPO are clearly better.** They keep the actual question ("what would you dare me to do for a year…"), which SFT drops; SFT just repeats "I'm not sure what to do". Both copy the post almost word for word, and PPO is truncated.

## Sample 3 (eval post 21; picked because: reward model prefers PPO/DPO most over SFT)

```
SUBREDDIT: r/relationships
TITLE: Me [28 M] with my GF [28 F] of 3 months, she lives an hour away and I have conflicting feelings depending on whether I'm with her or not
POST: I posted on here a while back about making things official with a girl I'd been seeing. We ended up doing so, and it's been going alright. However, she lives about about an hour away from, so we only see each other ~3 days/week. I've noticed that I feel differently about her depending on whether I'm with her or not. When I'm not with her, I tend to focus on her negative attributes and ruminate about the things I don't like about her -- not only the things that bug and annoy me, but how much better it would be to be single again. However, when I'm with her, I tend to forget those things to some degree and I do enjoy her company. Has anyone else felt this way/Is this normal? If so, how do you manage? 

And might this be indicative of some kind of incompatibility?

I know the easy answer to this dilemma is to just 'hang out with her more,' but that's not really possible given my work and her schedule.
TL;DR:
```

- **Human reference:** Pseudo LDR -- having conflicting feelings about my GF depending on whether I'm physically with her or not -- not really sure what to do about it or if it's indicative of anything/an ominous sign.
- **SFT** (r_phi -0.67, 28 tokens): I feel like I'm with my girlfriend more than I am with her, but I don't know if this is normal or not.
- **PPO** (r_phi +3.06, 35 tokens): Girl I've been seeing for ~3 months, she lives an hour away and I have conflicting feelings depending on whether I'm with her or not. Is this normal?
- **DPO** (r_phi +4.01, 47 tokens): Girl I've been seeing lives an hour away and I tend to focus on her negative attributes and ruminate about the things I don't like about her. Is this normal? Is this indicative of some kind of incompatibility?

**Why:** **PPO is clearly better.** It matches the reference: GF of ~3 months an hour away, conflicting feelings depending on whether he is with her, "is this normal?". SFT's summary is self-contradictory. DPO is good but adds a question the poster did not ask ("incompatibility?").

## Sample 4 (eval post 194; picked because: reward model prefers SFT most over PPO/DPO)

```
SUBREDDIT: r/tifu
TITLE: TIFU by forgetting to register my dog
POST: Earlier this year I finally was in a situation to get a dog, after a long search I found a beautiful - if somewhat dimwitted - Collie x Huntaway named Cooper. I got him as throughout my childhood my family had dogs, one in particular, a Collie name Lofty, was my best friend throughout my teenage year, but I left home at 16 and my family moved,so I never saw him again. 

Anyway, so the last couple of weeks I've been working heaps of hours to try and afford a decent Christmas this year, and while I normally work at home (I'm a programmer, unfortunately not a well paid one) Ive been working on site with my main client testing code for the last few days. I got home yesterday with a note from my neighbours complaining that my dog had been barking all day, I was surprised, but I figured since I only have one more day of work for the year I'd sort that out over my Christmas break. 

So I get home today and my daughter pulls a note out of the letter box, it's from our local council and basically says that there was a noise complaint, however they don't have a dog registered at our address, so we need to contact them in 24 hours to sort it or they'll take my dog. 

It's at this moment I realise that I completely forgot to register him. I got him microchipped the day I got him and had spent a fortune getting his vaccinations up to date as his old owners neglected him. After all that I didn't have the $100 to register him that month so I though I'd wait until next pay. Fast forward 6 months and I'd completely forgotten. 

Now I have to pay $400 in fines (aka everything I'd saved for presents) or have my dog taken away, more fines issues and him likely put to sleep. All on the 23rd of December.
TL;DR:
```

- **Human reference:** forgot to register dog, now have to spend all my money I saved for Christmas presents or my dog will be taken away.
- **SFT** (r_phi +1.66, 36 tokens): I forgot to register my dog, he was registered at my address, and now I have to pay $400 in fines and have to get him put to sleep.
- **PPO** (r_phi -0.85, 48 tokens): forgot to register my dog, got it microchipped, got it registered at address, got it registered at 24 hours after I got it. Now I've got to pay $400 in fines and have my dog taken
- **DPO** (r_phi +1.47, 38 tokens): Dog barking all day, got him microchipped, had spent a fortune getting him registered, forgot to register him, got him microchipped, got fined, dog taken away.

**Why:** **PPO and DPO are clearly worse.** SFT is a clean one-sentence summary. PPO repeats "got it registered" three times and is cut off; DPO is contradictory ("spent a fortune getting him registered, forgot to register him") and invents barking. Longer, detail-copying output with repetition — the length bias at work.

## Sample 5 (eval post 116; picked because: reward model prefers SFT most over PPO/DPO)

```
SUBREDDIT: r/Advice
TITLE: Neighbor keeps telling the cops my car is abandoned
POST: My wife and I live in a quiet little neighborhood in a quaint Chicago suburb (Plainfield). We have an 18month old and a baby on the way due in early May. We have 2 Honda accords - small for a growing family. I recently bought my wife a Toyota Highlander for more space. I'm planning to sell one of the Accords but haven't had a chance yet due to everything going on at home and work as well as the weather. 

We live in a townhouse that's one of four connected units. 

I have the Accord I plan to sell parked in a public spot about 50 feet from my unit. The problem is that one of my neighbors (I suspect the crabby old lady that has called the police on us before for parking too close to her house and whose unit is directly near the spot I parked in) keeps calling the police and claiming that my car's an abandoned vehicle. Nobody has asked us to move out car. We live in a little parkway area and it's just 8 townhouses (two 4-unit structures) that share 4 public parking spots...which are usually empty because everyone parks in their garages or driveways. Everyone in our little parkway knows our cars.

Both times the police came they left without taking any action. The car is obviously not abandoned and I move it every few days or so. There is nothing illegal about my use of the space as its a public spot. There are three other spots still empty every day. My concern is one of these days an officer may come when I'm not home and will tag my car with a tow notice because all he/she knows is that somebody called in an abandoned vehicle. How do I get the crabby old woman (I'm positive it's her...she's the only neighbor that's unfriendly and rude) to leave us alone and stop calling in our car which she definitely knows is not abandoned?
TL;DR:
```

- **Human reference:** my neighbor knows my car but keeps calling it in as an abandoned vehicle. Cops just leave without taking action because its obviously not an abandoned vehicle. How do I get the suspected neighbor to leave me alone?
- **SFT** (r_phi +2.58, 29 tokens): Neighbor keeps calling the cops on us and claims our car is an abandoned vehicle. How do I get her to leave us alone?
- **PPO** (r_phi +0.25, 48 tokens): Neighbor keeps calling the cops on us about parking too close to our unit. We live in a little parkway area and it's just 8 townhouses (two 4-unit structures) that share 4 public parking spots
- **DPO** (r_phi +1.91, 48 tokens): Neighbor keeps calling the cops on us when we park in public parking spots. We live in a quiet little parkway area. How do I get the crabby old woman (I'm positive it's her) to leave us alone

**Why:** **PPO is clearly worse.** SFT is almost the human reference (abandoned-car claim + "how do I get her to leave us alone?"). PPO drifts into irrelevant parking-lot details, loses both the claim and the question, and is truncated. DPO keeps the question but adds wrong details and is also truncated.

## Sample 6 (eval post 265; picked because: reward model prefers SFT most over PPO/DPO)

```
SUBREDDIT: r/AskReddit
TITLE: A conservative kid I went to HS with posted this, but I'm not sure as to the validity of his statement...
POST: I took Micro and Macro Econ in College, I study up on current politics pretty much everyday but something about this quote just seemed inaccurate.  Maybe my fellow redditors can shed some light or prove actual specific data either supporting or going against said quote.
 
"There comes a moment when the population realizes that it has to stop, and sometimes it takes a form of tax the rich people, which is a reflection more of the lack of understanding of how the economy works. Rich people are now being defined by the administration as people who make $1 million. Well, most of the businesses in America, other than giant corporations, are paying taxes under Chapter S, partnerships or individual proprietorships. So somebody shows that they've made $3 million or $2 million this year and they paid personal taxes on that money. They subtract the cost of living and then what's left after, and that does not show that probably 25% or 30% of their profits are tied up in accounts receivable or inventory, stuff that they can't spend or get their hands on, but to support their business and their employment. And then they take whatever is left, these so-called millionaires, and they open up another shop or another office. And that, that is the only known engine of growth in the United States of America. And we have an administration that is fanning the fires that this is somehow undeserved, profligate millionaires, and it is worse than hypocrisy. It is totally dishonest. It represents by young people who don't know the difference, simple misunderstanding and the lack of understanding of how the economy works or what's going on in America, but if it's politician that does it or union leader, then it represents something much more pernicious. It represents a deliberate misleading of the public." - Steve Wynn
TL;DR:
```

- **Human reference:** Quote about why millionaires shouldn't be taxed because most are small business owners who reinvest their gains.  We, as young people, clearly don't understand how the economy works and will ruin it all.
- **SFT** (r_phi +2.16, 42 tokens): Conservative kid posted this, but I'm not sure as to the validity of his statement.  Maybe my fellow redditors can shed some light or prove specific data either supporting or going against said quote.
- **PPO** (r_phi +2.25, 43 tokens): Conservative kid posted this quote, but I'm not sure as to the validity of his statement.  Maybe my fellow reddit can shed some light or prove actual specific data either supporting or going against said quote.
- **DPO** (r_phi +0.52, 33 tokens): Conservative kid posted this quote, but I'm not sure as to the validity of his statement, and I'm not sure as to the validity of his statement.

**Why:** **DPO is clearly worse.** It repeats the same clause twice and says nothing about the quote. SFT and PPO are nearly identical and both just copy the poster's opening sentences — none of the three summarises what the quote actually argues.

## Sample 7 (eval post 19; picked because: random)

```
SUBREDDIT: r/AskReddit
TITLE: Reddit, whats the weirdest/creepiest thing that ever happened to you following a break up?
POST: I'll go first.
So I'm going out with this girl for about a year and a half, it started off as just fooling around but things got pretty serious fast. Fast forward about two months ago and one of my close friends tell me that he seen her out with some guy at the cinema, at first I thought it couldn't of been her because she had told me that she was spending the week at her moms the night my friend saw her. I try texting and calling her all week but doesn't answer any of it.
When she gets back I immediately ask her was she cheating on me. Not the best approach I know but I was frustrated as hell. She starts bursting into tears and tells me its all true and that she spent the week at this guys place, so i do what any normal man would do in that situation and I lose my shit. We argue for what must have been an hour and we end up breaking up. I clearly remember calling her a whore when she was walking out the door, not my most mature moment.

This is when it gets weird, about two weeks ago I get a text saying 'You're dead' off an unknown number. I ignore it and then just two or three days after that I'm out with friends, just hanging out and this random guy comes up to us from across the road and punches me in the face. Now I'm kind of a small guy so I drop like a ton of bricks. As the guys runs away he yells back 'Thats for calling my girlfriend a whore'. Turns out my ex got with the guy she was cheating with me with and told him what i called and her. I rang my ex asking for an explanation and she told me 'Thats what you get for messing with his girl'.

So Reddit now its your turn.
TL;DR:
```

- **Human reference:** Girlfriend cheated on me, broke up, called her a whore, she gets her new boyfriend(who she cheated on me with) to punch me in the face for calling her a whore.
- **SFT** (r_phi +0.16, 37 tokens): Ex got with a guy he was cheating with me with, told him what i called and her, and now I'm a small guy and I'm a little pissed.
- **PPO** (r_phi -0.07, 31 tokens): Ex got with a guy at the cinema, punched me in the face, and now I'm a small guy and can't handle it anymore.
- **DPO** (r_phi +0.57, 30 tokens): Ex got with a guy she was cheating with me with and told him what i called and her. I ignored it and now its your turn.

**Why:** **No clear winner; all three are poor.** PPO is the only one to mention the key event (getting punched), but attributes it wrongly; SFT and DPO are garbled, and DPO ends with "now its your turn" from the post's AskReddit framing.

## Sample 8 (eval post 146; picked because: random)

```
SUBREDDIT: r/relationship_advice
TITLE: Am I overreacting?
POST: My boyfriend is going on a trip to visit a friend who is currently overseas. She is already there, and will meet up with him when he arrives. They have been good friends since before we dated, and I found out they had a relationship (again, before I knew him). He once told me he had very strong feelings for her, second only to me. 

I have no problem with the visiting, but I asked that they don't share a room just the two of them. I wouldn't mind them being in a large, dorm style room, but otherwise prefer they sleep separately. She got upset and asked if he was joking, saying it was going to cost more money etc. He stuck firm (maybe because I was there) and said no. I really don't want to come between them, but I am incredibly uncomfortable with the thought of someone that's had sex with him sleeping with him. It is not a matter of trust, as he once cheated on me before, and I seriously doubt he would do it again.
TL;DR:
```

- **Human reference:** my boyfriend is going overseas to stay with an exgirlfriend (now platonic friends). I asked them to sleep in separate rooms as I wasn't comfortable with them sleeping next to each other. Am I overreacting?
- **SFT** (r_phi +0.41, 25 tokens): Boyfriend is going on a trip to visit a friend, and I am uncomfortable with the idea of them sleeping together.
- **PPO** (r_phi +1.32, 48 tokens): Boyfriend is going on a trip to visit friend, and will meet up with him when he arrives. They have been good friends since before we dated. They have been good friends since before we dated. Should I be concerned about this?
- **DPO** (r_phi +2.48, 45 tokens): My boyfriend is going on a trip to visit a friend who is currently overseas. They have been good friends since before we dated. They have a relationship, but I am uncomfortable with the idea of them sleeping together.

**Why:** **DPO is better than SFT.** It covers the trip, the past relationship and the discomfort about sleeping arrangements. SFT is correct but thin and misses the "am I overreacting?" question; PPO repeats a sentence and is cut off.

## Sample 9 (eval post 73; picked because: random)

```
SUBREDDIT: r/relationships
TITLE: I [16 M] won't be with my Girlfriend [16 F] of 5 months, for valentines day and I'm feeling depressed
POST: So about 3 weeks ago I got caught trying to sneak over to my girlfriend's house when her parents weren't home. I'll admit, I wanted to have sex (we've done it twice before) and just have some alone time together with her for a while. We live about 45 minutes away from each other and neither of us have our licence so we usually depend on our friends, or our parents to take us from house to house

Any way long story short we got caught. She got grounded for a month and I got grounded till "I don't know when". I'm missing a lot of things because of this and I don't know when It's going to stop. 

We had a Mardi gras balle (Yes, I live in Louisiana) at school two weeks after and my girlfriend got court! She was allowed to attend and I was not. Come to find out she got voted queen but I still felt terrible because she had to go without a date because of my actions. 

Now it's valentines day and I made her an awesome present she won't get to see on valentines day because I am forbidden to see her at all. Not to mention I got my phone taken away so now I have no way to talk to her at all! I feel like I disappointed her and it's killing me inside.
TL;DR:
```

- **Human reference:** Can't see or talk to my girlfriend on valentines day and I feel depressed. I don't know what to do.
- **SFT** (r_phi -1.48, 33 tokens): I'm feeling depressed about my girlfriend and I don't know what to do. I'm not sure if I should just move on or try to fix it.
- **PPO** (r_phi +1.85, 37 tokens): got caught trying to sneak over to girlfriend's house while her parents weren't home, got caught and now I won't be able to see her at all. Feeling depressed inside.
- **DPO** (r_phi -0.26, 48 tokens): I got caught trying to sneak over to my girlfriend's house when her parents weren't home, got caught and got grounded for a month, won't be seeing her on valentines day. I feel like I disappointed her and it's

**Why:** **PPO is better.** It gives the cause (caught sneaking over, can't see her) and the feeling, close to the reference. SFT is generic and has no event; DPO has the most detail but is cut off mid-sentence.

## Sample 10 (eval post 236; picked because: random)

```
SUBREDDIT: r/relationships
TITLE: My girlfriend [16/F] broke up with me [17/M] last night
POST: We were getting along great for three weeks and then she suddenly said she couldn't handle a relationship right now and she wants to make herself happy. She was my first everything...first kiss, first dance, all of that... and then she just left. She said she didn't want to lose me and she still likes me but she needs to focus on herself. 

And honestly, I don't want to lose her either. We both went through a lot of pain together with other people and we just kind of wound up with each other, and I still really like her. 

She, and everyone else, keeps saying there's nothing wrong with me but there clearly is. Before her I was getting pretty close with another girl, but she decided she couldn't see being in a relationship with me. And now my ex suddenly decides she doesn't want to be in a relationship with me.

I think my problem is that I'm too moody because I get into times when I'm really happy and times when I get really depressed, regardless of other people. The breakup did not really help. 

I don't really know what I'm expecting with this, but I right now I just kind of want to go away and not feel anything like I used to.
TL;DR:
```

- **Human reference:** Girlfriend broke up with me... I don't know if I should still keep fighting for her and for myself or if I should just let myself go.
- **SFT** (r_phi -0.89, 25 tokens): Girlfriend broke up with me, and now I'm not sure if I'm going to be able to handle it.
- **PPO** (r_phi +0.87, 41 tokens): Girlfriend broke up with me last night. She still likes me but she needs to focus on herself. I don't want to lose her. What can I do to make this easier for me?
- **DPO** (r_phi +0.35, 37 tokens): Girlfriend broke up with me last night. She wants to focus on herself and not lose me. I feel like I'm too depressed and moody. What should I do?

**Why:** **PPO and DPO are better.** Both include that she broke up to focus on herself and end with a question, like the reference. SFT is generic ("not sure if I'm going to be able to handle it").

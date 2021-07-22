# k2-speechbrain
In this repository, I try to combine k2 with speechbrain to decoding well and fastly.

**Notice:**, I just did a preliminary explore about integrating k2 into speechbrain. And there is still a big space to improve it if anyone is interested in it.

At the basis of the [codes](https://gist.github.com/csukuangfj/c68697cd144c8f063cc7ec4fd885fd6f) from csukuangfj (thank him!), I try to combine k2 with the pretrained transformer encoder from speechbrain and get some results on LibriSpeech. I use the public pretrained transformer encoder from the speechbrain team. I test the two datasets' samples (test-clean and test-other) one by one. And in my experiments, I found that the decoding process based on k2 was much faster than speechbrain (transformer-LM).

Some results I get are as follows (WER and Duration, based on 1 GPU):
``` 
                             Method                             |  test-clean(WER%) |  test-clean (h:m:s)  |  test-other(WER%)
------------------------------------------------------------------------------------------------------------------------------
    speechbrain (public, lm_weight=0.6, ctc_weight=0.52, bs=66) |       2.46        |            /         |       5.77
------------------------------------------------------------------------------------------------------------------------------
 speechbrain (reproduce, lm_weight=0.6, ctc_weight=0.52, bs=66) |       2.52        |        02:33:18      |       5.93
------------------------------------------------------------------------------------------------------------------------------
  speechbrain (reproduce, lm_weight=0.0, ctc_weight=0.0, bs=1)  |       4.43        |        00:11:26      |       10.01
------------------------------------------------------------------------------------------------------------------------------
                pre-encoder-output+softmax+greedy               |       17.42       |        00:02:30      |       23.38
------------------------------------------------------------------------------------------------------------------------------
                  k2_ctc_topo+pre-encoder (bs=8)                |       5.88        |        00:14:00      |       13.82
------------------------------------------------------------------------------------------------------------------------------
         k2_HLG+pre-encoder (use-whole-lattices=False)          |       8.49        |        00:04:31      |       17.42
------------------------------------------------------------------------------------------------------------------------------
   k2_HLG+pre-encoder (use-whole-lattices=True, lm-scale=1.2)   |     6.39 (cpu)    |         / (cpu)      |       16.67 (cpu)
------------------------------------------------------------------------------------------------------------------------------ 
   k2_HLG+pre-encoder (use-whole-lattices=True, lm-scale=1.2)   |     9.05 (gpu)    |        00:12:54      |       17.68    
------------------------------------------------------------------------------------------------------------------------------ 
````
<!-- Decoding duration (based on 1 GPU):
``` 
                             Method                             |  test-clean (h:m:s) 
--------------------------------------------------------------------------------------
    speechbrain (public, lm_weight=0.6, ctc_weight=0.52, bs=66) |        /        
--------------------------------------------------------------------------------------
    speechbrain (public, lm_weight=0.0, ctc_weight=0.0, bs=1)   |      00:11:26      
--------------------------------------------------------------------------------------
                pre-encoder-output+softmax+greedy               |      00:02:30      
--------------------------------------------------------------------------------------
                  k2_ctc_topo+pre-encoder (bs=8)                |      00:14:00     
--------------------------------------------------------------------------------------
         k2_HLG+pre-encoder (use-whole-lattices=False)          |      00:04:31    
--------------------------------------------------------------------------------------
   k2_HLG+pre-encoder (use-whole-lattices=True, lm-scale=1.2)   |      / (cpu)
--------------------------------------------------------------------------------------       
   k2_HLG+pre-encoder (use-whole-lattices=True, lm-scale=1.2)   |      00:12:54   
-------------------------------------------------------------------------------------- 
````
 -->
Results based on different lm-scale (k2_HLG+pre-encoder, use-whole-lattices=True, one GPU):
```
     lm-scale    |  test-clean (WER%) |  test-clean (h:m:s)
--------------------------------------------------------------
        0.0      |        40.57       |      00:12:25
--------------------------------------------------------------
        0.1      |        4.84        |      00:12:10
--------------------------------------------------------------
        0.2      |        4.68        |      00:12:06
--------------------------------------------------------------
        0.3      |        4.62        |      00:12:01
--------------------------------------------------------------
        0.4      |        4.71        |      00:12:15
--------------------------------------------------------------
        0.5      |        4.84        |      00:12:06
--------------------------------------------------------------
        0.6      |        5.09        |      00:12:18
--------------------------------------------------------------
        0.7      |        5.45        |      00:12:06
--------------------------------------------------------------
        0.8      |        5.91        |      00:12:30
--------------------------------------------------------------
        0.9      |        6.48        |      00:12:15
--------------------------------------------------------------
        1.0      |        7.28        |      00:12:03
--------------------------------------------------------------
        1.1      |        8.07        |      00:12:29
--------------------------------------------------------------
        1.2      |        9.05        |      00:12:54
--------------------------------------------------------------
```
How to run:
```
bash run.sh
```

Some decoding results: ([all-results](https://drive.google.com/drive/folders/1s1dWtfgBvyziakuNhf4L7QmGglXRv2Ig?usp=sharing))
```
%WER 6.39 [ 3359 / 52576, 132 ins, 1571 del, 1656 sub ]
%SER 63.82 [ 1672 / 2620 ]
Scored 2620 sentences, 0 not present in hyp.
================================================================================
ALIGNMENTS

Format:
<utterance-id>, WER DETAILS
<eps> ; reference  ; on ; the ; first ;  line
  I   ;     S      ; =  ;  =  ;   S   ;   D  
 and  ; hypothesis ; on ; the ; third ; <eps>
================================================================================
672-122797-0033, %WER 0.00 [ 0 / 2, 0 ins, 0 del, 0 sub ]
A ; STORY
= ;   =  
A ; STORY
================================================================================
2094-142345-0041, %WER 100.00 [ 1 / 1, 0 ins, 0 del, 1 sub ]
DIRECTION
    S    
         
================================================================================
2830-3980-0026, %WER 100.00 [ 2 / 2, 0 ins, 1 del, 1 sub ]
VERSE ;  TWO 
  S   ;   D  
FIRST ; <eps>
================================================================================
5142-36377-0000, %WER 0.00 [ 0 / 13, 0 ins, 0 del, 0 sub ]
IT ; WAS ; ONE ; OF ; THE ; MASTERLY ; AND ; CHARMING ; STORIES ; OF ; DUMAS ; THE ; ELDER
=  ;  =  ;  =  ; =  ;  =  ;    =     ;  =  ;    =     ;    =    ; =  ;   =   ;  =  ;   =  
IT ; WAS ; ONE ; OF ; THE ; MASTERLY ; AND ; CHARMING ; STORIES ; OF ; DUMAS ; THE ; ELDER
================================================================================
6930-76324-0003, %WER 16.67 [ 2 / 12, 0 ins, 1 del, 1 sub ]
NOW ; WHAT ; WAS ; THE ; SENSE ; OF ; IT ; TWO ; INNOCENT ; BABIES ; LIKE ; THAT
 =  ;  =   ;  S  ;  =  ;   =   ; =  ; =  ;  =  ;    D     ;   =    ;  =   ;  =  
NOW ; WHAT ;  IS ; THE ; SENSE ; OF ; IT ; TWO ;  <eps>   ; BABIES ; LIKE ; THAT
================================================================================
260-123440-0007, %WER 0.00 [ 0 / 10, 0 ins, 0 del, 0 sub ]
I ; ALMOST ; THINK ; I ; CAN ; REMEMBER ; FEELING ; A ; LITTLE ; DIFFERENT
= ;   =    ;   =   ; = ;  =  ;    =     ;    =    ; = ;   =    ;     =    
I ; ALMOST ; THINK ; I ; CAN ; REMEMBER ; FEELING ; A ; LITTLE ; DIFFERENT
================================================================================
7021-79730-0003, %WER 12.90 [ 8 / 62, 0 ins, 3 del, 5 sub ]
AS ; THE ; CHAISE ; DRIVES ; AWAY ; MARY ; STANDS ; BEWILDERED ; AND ; PERPLEXED ; ON ; THE ;   DOOR   ;  STEP ; HER ; MIND ; IN ; A ; TUMULT ; OF ; EXCITEMENT ; IN ; WHICH ; HATRED ; OF ; THE ; DOCTOR ; DISTRUST ; AND ; SUSPICION ; OF ; HER ; MOTHER ; DISAPPOINTMENT ; VEXATION ; AND ; ILL ; HUMOR  ; SURGE ; AND ; SWELL ; AMONG ; THOSE ; DELICATE ; ORGANIZATIONS ; ON ; WHICH ; THE ; STRUCTURE ; AND ; DEVELOPMENT ; OF ; THE ; SOUL ; SO ; CLOSELY ; DEPEND ; DOING  ; PERHAPS ;   AN  ; IRREPARABLE ; INJURY
=  ;  =  ;   S    ;   =    ;  =   ;  =   ;   =    ;     =      ;  =  ;     =     ; =  ;  =  ;    S     ;   D   ;  =  ;  =   ; =  ; = ;   =    ; =  ;     =      ; =  ;   =   ;   =    ; =  ;  =  ;   =    ;    =     ;  =  ;     =     ; =  ;  =  ;   =    ;       =        ;    =     ;  =  ;  =  ;   S    ;   =   ;  =  ;   =   ;   =   ;   =   ;    D     ;       =       ; =  ;   =   ;  =  ;     =     ;  =  ;      =      ; =  ;  =  ;  =   ; =  ;    =    ;   =    ;   S    ;    S    ;   D   ;      =      ;   =   
AS ; THE ; CHASE  ; DRIVES ; AWAY ; MARY ; STANDS ; BEWILDERED ; AND ; PERPLEXED ; ON ; THE ; DOORSTEP ; <eps> ; HER ; MIND ; IN ; A ; TUMULT ; OF ; EXCITEMENT ; IN ; WHICH ; HATRED ; OF ; THE ; DOCTOR ; DISTRUST ; AND ; SUSPICION ; OF ; HER ; MOTHER ; DISAPPOINTMENT ; VEXATION ; AND ; ILL ; HUMOUR ; SURGE ; AND ; SWELL ; AMONG ; THOSE ;  <eps>   ; ORGANIZATIONS ; ON ; WHICH ; THE ; STRUCTURE ; AND ; DEVELOPMENT ; OF ; THE ; SOUL ; SO ; CLOSELY ; DEPEND ; DOINGS ;   AND   ; <eps> ; IRREPARABLE ; INJURY
================================================================================
1995-1836-0004, %WER 6.25 [ 6 / 96, 1 ins, 0 del, 5 sub ]
AS ; SHE ; AWAITED ; HER ; GUESTS ; SHE ; SURVEYED ; THE ; TABLE ; WITH ; BOTH ; SATISFACTION ; AND ; DISQUIETUDE ; FOR ; HER ; SOCIAL ; FUNCTIONS ; WERE ; FEW ; TONIGHT ; THERE ; WERE ; SHE ; CHECKED ; THEM ; OFF ; ON ; HER ; FINGERS ; SIR ; JAMES ; CREIGHTON ; THE ; RICH ; ENGLISH ; MANUFACTURER ; AND ; LADY ; CREIGHTON ; MISTER ; AND ; MISSUS ; VANDERPOOL ; <eps> ; MISTER ; HARRY ; CRESSWELL ; AND ; HIS ; SISTER ; JOHN ; TAYLOR ; AND ; HIS ; SISTER ; AND ; MISTER ; CHARLES ; SMITH ; WHOM ; THE ; EVENING ; PAPERS ; MENTIONED ; AS ; LIKELY ; TO ; BE ; UNITED ; STATES ; SENATOR ; FROM ; NEW ; JERSEY ; A ; SELECTION ; OF ; GUESTS ; THAT ; HAD ; BEEN ; DETERMINED ; UNKNOWN ; TO ; THE ; HOSTESS ; BY ; THE ; MEETING ; OF ; COTTON ; INTERESTS ; EARLIER ; IN ; THE ; DAY
=  ;  =  ;    =    ;  =  ;   S    ;  =  ;    =     ;  =  ;   =   ;  =   ;  =   ;      =       ;  =  ;      =      ;  =  ;  =  ;   =    ;     =     ;  =   ;  =  ;    =    ;   =   ;  =   ;  =  ;    =    ;  =   ;  =  ; =  ;  =  ;    =    ;  =  ;   =   ;     S     ;  =  ;  =   ;    =    ;      =       ;  =  ;  =   ;     S     ;   =    ;  =  ;   =    ;     S      ;   I   ;   =    ;   S   ;     =     ;  =  ;  =  ;   =    ;  =   ;   =    ;  =  ;  =  ;   =    ;  =  ;   =    ;    =    ;   =   ;  =   ;  =  ;    =    ;   =    ;     =     ; =  ;   =    ; =  ; =  ;   =    ;   =    ;    =    ;  =   ;  =  ;   =    ; = ;     =     ; =  ;   =    ;  =   ;  =  ;  =   ;     =      ;    =    ; =  ;  =  ;    =    ; =  ;  =  ;    =    ; =  ;   =    ;     =     ;    =    ; =  ;  =  ;  = 
AS ; SHE ; AWAITED ; HER ; GUEST  ; SHE ; SURVEYED ; THE ; TABLE ; WITH ; BOTH ; SATISFACTION ; AND ; DISQUIETUDE ; FOR ; HER ; SOCIAL ; FUNCTIONS ; WERE ; FEW ; TONIGHT ; THERE ; WERE ; SHE ; CHECKED ; THEM ; OFF ; ON ; HER ; FINGERS ; SIR ; JAMES ;   WALTON  ; THE ; RICH ; ENGLISH ; MANUFACTURER ; AND ; LADY ;    CAR    ; MISTER ; AND ; MISSUS ;    VAN     ;   PO  ; MISTER ; HENRY ; CRESSWELL ; AND ; HIS ; SISTER ; JOHN ; TAYLOR ; AND ; HIS ; SISTER ; AND ; MISTER ; CHARLES ; SMITH ; WHOM ; THE ; EVENING ; PAPERS ; MENTIONED ; AS ; LIKELY ; TO ; BE ; UNITED ; STATES ; SENATOR ; FROM ; NEW ; JERSEY ; A ; SELECTION ; OF ; GUESTS ; THAT ; HAD ; BEEN ; DETERMINED ; UNKNOWN ; TO ; THE ; HOSTESS ; BY ; THE ; MEETING ; OF ; COTTON ; INTERESTS ; EARLIER ; IN ; THE ; DAY
================================================================================
4507-16021-0047, %WER 10.39 [ 8 / 77, 0 ins, 7 del, 1 sub ]
YESTERDAY ; YOU ; WERE ; TREMBLING ; FOR ; A ; HEALTH ; THAT ; IS ; DEAR ; TO ; YOU ; TO ; DAY ; YOU ; FEAR ;  FOR  ; YOUR ; OWN ; TO ; MORROW ; IT ; WILL ; BE ; ANXIETY ; ABOUT ; MONEY ; THE ; DAY ; AFTER ; TO ; MORROW ; THE ; DIATRIBE ; OF ; A ; SLANDERER ; THE ; DAY ; AFTER ; THAT ; THE ; MISFORTUNE ; OF ; SOME ; FRIEND ; THEN ; THE ; PREVAILING ; WEATHER ; THEN ; SOMETHING ; THAT ; HAS ; BEEN ; BROKEN ; OR ; LOST ; THEN ; A ; PLEASURE ; WITH ; WHICH ; YOUR ; CONSCIENCE ;  AND  ;  YOUR ; VERTEBRAL ; COLUMN ; REPROACH ; YOU ; AGAIN ; THE ; COURSE ; OF ; PUBLIC ; AFFAIRS
    =     ;  =  ;  =   ;     =     ;  =  ; = ;   D    ;  =   ; =  ;  =   ; =  ;  =  ; =  ;  =  ;  =  ;  =   ;   D   ;  =   ;  =  ; =  ;   =    ; =  ;  =   ; =  ;    =    ;   =   ;   =   ;  =  ;  =  ;   =   ; =  ;   =    ;  =  ;    S     ; =  ; = ;     =     ;  =  ;  =  ;   =   ;  =   ;  =  ;     =      ; =  ;  =   ;   =    ;  =   ;  =  ;     =      ;    =    ;  =   ;     =     ;  =   ;  =  ;  =   ;   =    ; =  ;  =   ;  =   ; = ;    =     ;  =   ;   =   ;  =   ;     =      ;   D   ;   D   ;     D     ;   D    ;    D     ;  =  ;   =   ;  =  ;   =    ; =  ;   =    ;    =   
YESTERDAY ; YOU ; WERE ; TREMBLING ; FOR ; A ; <eps>  ; THAT ; IS ; DEAR ; TO ; YOU ; TO ; DAY ; YOU ; FEAR ; <eps> ; YOUR ; OWN ; TO ; MORROW ; IT ; WILL ; BE ; ANXIETY ; ABOUT ; MONEY ; THE ; DAY ; AFTER ; TO ; MORROW ; THE ; DIETARY  ; OF ; A ; SLANDERER ; THE ; DAY ; AFTER ; THAT ; THE ; MISFORTUNE ; OF ; SOME ; FRIEND ; THEN ; THE ; PREVAILING ; WEATHER ; THEN ; SOMETHING ; THAT ; HAS ; BEEN ; BROKEN ; OR ; LOST ; THEN ; A ; PLEASURE ; WITH ; WHICH ; YOUR ; CONSCIENCE ; <eps> ; <eps> ;   <eps>   ; <eps>  ;  <eps>   ; YOU ; AGAIN ; THE ; COURSE ; OF ; PUBLIC ; AFFAIRS
```

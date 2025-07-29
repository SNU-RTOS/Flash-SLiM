import networkx as nx
import matplotlib.pyplot as plt

def parse_node_details(node_details):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Dictionary to store node details
    node_info = {}
    
    # Parse each node
    for node_block in node_details.split('  Node ')[1:]:
        lines = node_block.strip().split('\n')
        
        # Extract node index and operator
        node_header = lines[0].split(':')
        node_idx = node_header[0].strip()
        operator = node_header[1].strip()
        
        # Parse input, output, intermediate, and temporary tensors
        inputs = []
        outputs = []
        intermediates = []
        temporaries = []
        
        for line in lines[1:]:
            if 'Input Tensors:' in line:
                inputs = [item.split(':')[1].strip() for item in lines[lines.index(line)+1:lines.index(line)+1+len([l for l in lines[lines.index(line)+1:] if 'Output' not in l and ':' in l])]]
            elif 'Output Tensors:' in line:
                outputs = [item.split(':')[1].strip() for item in lines[lines.index(line)+1:lines.index(line)+1+len([l for l in lines[lines.index(line)+1:] if 'Intermediate' not in l and ':' in l])]]
            elif 'Intermediate Tensors:' in line:
                intermediates = [item.split(':')[1].strip() for item in lines[lines.index(line)+1:lines.index(line)+1+len([l for l in lines[lines.index(line)+1:] if 'Temporary' not in l and ':' in l])]]
            elif 'Temporary Tensors:' in line:
                temporaries = [item.split(':')[1].strip() for item in lines[lines.index(line)+1:lines.index(line)+1+len([l for l in lines[lines.index(line)+1:] if ':' in l])]]
        
        # Store node details
        node_info[node_idx] = {
            'operator': operator,
            'inputs': inputs,
            'outputs': outputs,
            'intermediates': intermediates,
            'temporaries': temporaries
        }
        
        # Add nodes to the graph
        G.add_node(node_idx, label=f"{node_idx}\n{operator}")
        
        # Add edges from input tensors to this node
        for input_tensor in inputs:
            for other_node, details in node_info.items():
                if input_tensor in details['outputs']:
                    G.add_edge(other_node, node_idx)
                    break
    
    return G, node_info

# Create the graph
node_details = '''=== Node Details ===
  Node 0:
    Operator: RESHAPE
    Input Tensors:
      Input 0: 39
      Input 1: 276
    Output Tensors:
      Output 0: 284
    Intermediate Tensors:
    Temporary Tensors:
  Node 1:
    Operator: EMBEDDING_LOOKUP
    Input Tensors:
      Input 0: 284
      Input 1: 283
    Output Tensors:
      Output 0: 285
    Intermediate Tensors:
    Temporary Tensors:
  Node 3:
    Operator: CAST
    Input Tensors:
      Input 0: 5
    Output Tensors:
      Output 0: 287
    Intermediate Tensors:
    Temporary Tensors:
  Node 8:
    Operator: LESS
    Input Tensors:
      Input 0: 5
      Input 1: 277
    Output Tensors:
      Output 0: 292
    Intermediate Tensors:
    Temporary Tensors:
  Node 9:
    Operator: Custom Operator
    Input Tensors:
      Input 0: 5
      Input 1: 278
    Output Tensors:
      Output 0: 293
    Intermediate Tensors:
    Temporary Tensors:
  Node 10:
    Operator: SELECT
    Input Tensors:
      Input 0: 292
      Input 1: 293
      Input 2: 5
    Output Tensors:
      Output 0: 294
    Intermediate Tensors:
    Temporary Tensors:
  Node 11:
    Operator: RESHAPE
    Input Tensors:
      Input 0: 294
      Input 1: 273
    Output Tensors:
      Output 0: 295
    Intermediate Tensors:
    Temporary Tensors:
  Node 12:
    Operator: CAST
    Input Tensors:
      Input 0: 295
    Output Tensors:
      Output 0: 296
    Intermediate Tensors:
    Temporary Tensors:
  Node 13:
    Operator: GREATER_EQUAL
    Input Tensors:
      Input 0: 296
      Input 1: 279
    Output Tensors:
      Output 0: 297
    Intermediate Tensors:
    Temporary Tensors:
  Node 14:
    Operator: LESS_EQUAL
    Input Tensors:
      Input 0: 296
      Input 1: 280
    Output Tensors:
      Output 0: 298
    Intermediate Tensors:
    Temporary Tensors:
  Node 15:
    Operator: LOGICAL_AND
    Input Tensors:
      Input 0: 297
      Input 1: 298
    Output Tensors:
      Output 0: 299
    Intermediate Tensors:
    Temporary Tensors:
  Node 16:
    Operator: REDUCE_ALL
    Input Tensors:
      Input 0: 299
      Input 1: 276
    Output Tensors:
      Output 0: 300
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1826
      Temporary 1: 1827
      Temporary 2: 1828
      Temporary 3: 1829
  Node 17:
    Operator: GATHER_ND
    Input Tensors:
      Input 0: 264
      Input 1: 296
    Output Tensors:
      Output 0: 301
    Intermediate Tensors:
    Temporary Tensors:
  Node 18:
    Operator: RESHAPE
    Input Tensors:
      Input 0: 300
      Input 1: 275
    Output Tensors:
      Output 0: 302
    Intermediate Tensors:
    Temporary Tensors:
  Node 19:
    Operator: SELECT_V2
    Input Tensors:
      Input 0: 302
      Input 1: 301
      Input 2: 271
    Output Tensors:
      Output 0: 303
    Intermediate Tensors:
    Temporary Tensors:
  Node 54:
    Operator: RESHAPE
    Input Tensors:
      Input 0: 5
      Input 1: 265
    Output Tensors:
      Output 0: 338
    Intermediate Tensors:
    Temporary Tensors:
  Node 55:
    Operator: PACK
    Input Tensors:
      Input 0: 282
      Input 1: 338
      Input 2: 282
      Input 3: 282
    Output Tensors:
      Output 0: 339
    Intermediate Tensors:
    Temporary Tensors:
  Node 1542:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 272
      Input 1: 273
      Input 2: 274
      Input 3: 285
      Input 4: 287
    Output Tensors:
      Output 0: 289
      Output 1: 304
    Intermediate Tensors:
    Temporary Tensors:
  Node 6:
    Operator: COS
    Input Tensors:
      Input 0: 289
    Output Tensors:
      Output 0: 290
    Intermediate Tensors:
    Temporary Tensors:
  Node 7:
    Operator: SIN
    Input Tensors:
      Input 0: 289
    Output Tensors:
      Output 0: 291
    Intermediate Tensors:
    Temporary Tensors:
  Node 21:
    Operator: SUM
    Input Tensors:
      Input 0: 304
      Input 1: 206
    Output Tensors:
      Output 0: 305
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1830
      Temporary 1: 1831
      Temporary 2: 1832
      Temporary 3: 1833
  Node 1543:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 197
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 263
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 285
      Input 17: 290
      Input 18: 291
      Input 19: 305
    Output Tensors:
      Output 0: 315
      Output 1: 326
      Output 2: 337
    Intermediate Tensors:
    Temporary Tensors:
  Node 56:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 42
      Input 1: 337
      Input 2: 339
    Output Tensors:
      Output 0: 340
    Intermediate Tensors:
    Temporary Tensors:
  Node 57:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 19
      Input 1: 315
      Input 2: 339
    Output Tensors:
      Output 0: 341
    Intermediate Tensors:
    Temporary Tensors:
  Node 1544:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 196
      Input 1: 274
      Input 2: 285
      Input 3: 303
      Input 4: 326
      Input 5: 340
      Input 6: 341
    Output Tensors:
      Output 0: 345
      Output 1: 346
    Intermediate Tensors:
    Temporary Tensors:
  Node 63:
    Operator: SUM
    Input Tensors:
      Input 0: 346
      Input 1: 206
    Output Tensors:
      Output 0: 347
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1846
      Temporary 1: 1847
      Temporary 2: 1848
      Temporary 3: 1849
  Node 1545:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 193
      Input 1: 194
      Input 2: 195
      Input 3: 205
      Input 4: 262
      Input 5: 281
      Input 6: 345
      Input 7: 347
    Output Tensors:
      Output 0: 359
      Output 1: 360
    Intermediate Tensors:
    Temporary Tensors:
  Node 77:
    Operator: SUM
    Input Tensors:
      Input 0: 360
      Input 1: 206
    Output Tensors:
      Output 0: 361
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1868
      Temporary 1: 1869
      Temporary 2: 1870
      Temporary 3: 1871
  Node 1546:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 192
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 261
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 359
      Input 19: 361
    Output Tensors:
      Output 0: 371
      Output 1: 382
      Output 2: 393
    Intermediate Tensors:
    Temporary Tensors:
  Node 110:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 48
      Input 1: 393
      Input 2: 339
    Output Tensors:
      Output 0: 394
    Intermediate Tensors:
    Temporary Tensors:
  Node 111:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 38
      Input 1: 371
      Input 2: 339
    Output Tensors:
      Output 0: 395
    Intermediate Tensors:
    Temporary Tensors:
  Node 1547:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 191
      Input 1: 274
      Input 2: 303
      Input 3: 359
      Input 4: 382
      Input 5: 394
      Input 6: 395
    Output Tensors:
      Output 0: 399
      Output 1: 400
    Intermediate Tensors:
    Temporary Tensors:
  Node 117:
    Operator: SUM
    Input Tensors:
      Input 0: 400
      Input 1: 206
    Output Tensors:
      Output 0: 401
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1884
      Temporary 1: 1885
      Temporary 2: 1886
      Temporary 3: 1887
  Node 1548:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 188
      Input 1: 189
      Input 2: 190
      Input 3: 205
      Input 4: 260
      Input 5: 281
      Input 6: 399
      Input 7: 401
    Output Tensors:
      Output 0: 413
      Output 1: 414
    Intermediate Tensors:
    Temporary Tensors:
  Node 131:
    Operator: SUM
    Input Tensors:
      Input 0: 414
      Input 1: 206
    Output Tensors:
      Output 0: 415
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1906
      Temporary 1: 1907
      Temporary 2: 1908
      Temporary 3: 1909
  Node 1549:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 187
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 259
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 413
      Input 19: 415
    Output Tensors:
      Output 0: 425
      Output 1: 436
      Output 2: 447
    Intermediate Tensors:
    Temporary Tensors:
  Node 164:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 2
      Input 1: 447
      Input 2: 339
    Output Tensors:
      Output 0: 448
    Intermediate Tensors:
    Temporary Tensors:
  Node 165:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 47
      Input 1: 425
      Input 2: 339
    Output Tensors:
      Output 0: 449
    Intermediate Tensors:
    Temporary Tensors:
  Node 1550:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 186
      Input 1: 274
      Input 2: 303
      Input 3: 413
      Input 4: 436
      Input 5: 448
      Input 6: 449
    Output Tensors:
      Output 0: 453
      Output 1: 454
    Intermediate Tensors:
    Temporary Tensors:
  Node 171:
    Operator: SUM
    Input Tensors:
      Input 0: 454
      Input 1: 206
    Output Tensors:
      Output 0: 455
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1922
      Temporary 1: 1923
      Temporary 2: 1924
      Temporary 3: 1925
  Node 1551:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 183
      Input 1: 184
      Input 2: 185
      Input 3: 205
      Input 4: 258
      Input 5: 281
      Input 6: 453
      Input 7: 455
    Output Tensors:
      Output 0: 467
      Output 1: 468
    Intermediate Tensors:
    Temporary Tensors:
  Node 185:
    Operator: SUM
    Input Tensors:
      Input 0: 468
      Input 1: 206
    Output Tensors:
      Output 0: 469
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1944
      Temporary 1: 1945
      Temporary 2: 1946
      Temporary 3: 1947
  Node 1552:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 182
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 257
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 467
      Input 19: 469
    Output Tensors:
      Output 0: 479
      Output 1: 490
      Output 2: 501
    Intermediate Tensors:
    Temporary Tensors:
  Node 218:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 18
      Input 1: 501
      Input 2: 339
    Output Tensors:
      Output 0: 502
    Intermediate Tensors:
    Temporary Tensors:
  Node 219:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 1
      Input 1: 479
      Input 2: 339
    Output Tensors:
      Output 0: 503
    Intermediate Tensors:
    Temporary Tensors:
  Node 1553:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 181
      Input 1: 274
      Input 2: 303
      Input 3: 467
      Input 4: 490
      Input 5: 502
      Input 6: 503
    Output Tensors:
      Output 0: 507
      Output 1: 508
    Intermediate Tensors:
    Temporary Tensors:
  Node 225:
    Operator: SUM
    Input Tensors:
      Input 0: 508
      Input 1: 206
    Output Tensors:
      Output 0: 509
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1960
      Temporary 1: 1961
      Temporary 2: 1962
      Temporary 3: 1963
  Node 1554:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 178
      Input 1: 179
      Input 2: 180
      Input 3: 205
      Input 4: 256
      Input 5: 281
      Input 6: 507
      Input 7: 509
    Output Tensors:
      Output 0: 521
      Output 1: 522
    Intermediate Tensors:
    Temporary Tensors:
  Node 239:
    Operator: SUM
    Input Tensors:
      Input 0: 522
      Input 1: 206
    Output Tensors:
      Output 0: 523
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1982
      Temporary 1: 1983
      Temporary 2: 1984
      Temporary 3: 1985
  Node 1555:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 177
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 255
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 521
      Input 19: 523
    Output Tensors:
      Output 0: 533
      Output 1: 544
      Output 2: 555
    Intermediate Tensors:
    Temporary Tensors:
  Node 272:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 14
      Input 1: 555
      Input 2: 339
    Output Tensors:
      Output 0: 556
    Intermediate Tensors:
    Temporary Tensors:
  Node 273:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 51
      Input 1: 533
      Input 2: 339
    Output Tensors:
      Output 0: 557
    Intermediate Tensors:
    Temporary Tensors:
  Node 1556:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 176
      Input 1: 274
      Input 2: 303
      Input 3: 521
      Input 4: 544
      Input 5: 556
      Input 6: 557
    Output Tensors:
      Output 0: 561
      Output 1: 562
    Intermediate Tensors:
    Temporary Tensors:
  Node 279:
    Operator: SUM
    Input Tensors:
      Input 0: 562
      Input 1: 206
    Output Tensors:
      Output 0: 563
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 1998
      Temporary 1: 1999
      Temporary 2: 2000
      Temporary 3: 2001
  Node 1557:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 173
      Input 1: 174
      Input 2: 175
      Input 3: 205
      Input 4: 254
      Input 5: 281
      Input 6: 561
      Input 7: 563
    Output Tensors:
      Output 0: 575
      Output 1: 576
    Intermediate Tensors:
    Temporary Tensors:
  Node 293:
    Operator: SUM
    Input Tensors:
      Input 0: 576
      Input 1: 206
    Output Tensors:
      Output 0: 577
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2020
      Temporary 1: 2021
      Temporary 2: 2022
      Temporary 3: 2023
  Node 1558:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 172
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 253
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 575
      Input 19: 577
    Output Tensors:
      Output 0: 587
      Output 1: 598
      Output 2: 609
    Intermediate Tensors:
    Temporary Tensors:
  Node 326:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 54
      Input 1: 609
      Input 2: 339
    Output Tensors:
      Output 0: 610
    Intermediate Tensors:
    Temporary Tensors:
  Node 327:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 23
      Input 1: 587
      Input 2: 339
    Output Tensors:
      Output 0: 611
    Intermediate Tensors:
    Temporary Tensors:
  Node 1559:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 171
      Input 1: 274
      Input 2: 303
      Input 3: 575
      Input 4: 598
      Input 5: 610
      Input 6: 611
    Output Tensors:
      Output 0: 615
      Output 1: 616
    Intermediate Tensors:
    Temporary Tensors:
  Node 333:
    Operator: SUM
    Input Tensors:
      Input 0: 616
      Input 1: 206
    Output Tensors:
      Output 0: 617
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2036
      Temporary 1: 2037
      Temporary 2: 2038
      Temporary 3: 2039
  Node 1560:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 168
      Input 1: 169
      Input 2: 170
      Input 3: 205
      Input 4: 252
      Input 5: 281
      Input 6: 615
      Input 7: 617
    Output Tensors:
      Output 0: 629
      Output 1: 630
    Intermediate Tensors:
    Temporary Tensors:
  Node 347:
    Operator: SUM
    Input Tensors:
      Input 0: 630
      Input 1: 206
    Output Tensors:
      Output 0: 631
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2058
      Temporary 1: 2059
      Temporary 2: 2060
      Temporary 3: 2061
  Node 1561:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 167
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 251
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 629
      Input 19: 631
    Output Tensors:
      Output 0: 641
      Output 1: 652
      Output 2: 663
    Intermediate Tensors:
    Temporary Tensors:
  Node 380:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 24
      Input 1: 663
      Input 2: 339
    Output Tensors:
      Output 0: 664
    Intermediate Tensors:
    Temporary Tensors:
  Node 381:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 11
      Input 1: 641
      Input 2: 339
    Output Tensors:
      Output 0: 665
    Intermediate Tensors:
    Temporary Tensors:
  Node 1562:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 166
      Input 1: 274
      Input 2: 303
      Input 3: 629
      Input 4: 652
      Input 5: 664
      Input 6: 665
    Output Tensors:
      Output 0: 669
      Output 1: 670
    Intermediate Tensors:
    Temporary Tensors:
  Node 387:
    Operator: SUM
    Input Tensors:
      Input 0: 670
      Input 1: 206
    Output Tensors:
      Output 0: 671
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2074
      Temporary 1: 2075
      Temporary 2: 2076
      Temporary 3: 2077
  Node 1563:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 163
      Input 1: 164
      Input 2: 165
      Input 3: 205
      Input 4: 250
      Input 5: 281
      Input 6: 669
      Input 7: 671
    Output Tensors:
      Output 0: 683
      Output 1: 684
    Intermediate Tensors:
    Temporary Tensors:
  Node 401:
    Operator: SUM
    Input Tensors:
      Input 0: 684
      Input 1: 206
    Output Tensors:
      Output 0: 685
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2096
      Temporary 1: 2097
      Temporary 2: 2098
      Temporary 3: 2099
  Node 1564:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 162
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 249
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 683
      Input 19: 685
    Output Tensors:
      Output 0: 695
      Output 1: 706
      Output 2: 717
    Intermediate Tensors:
    Temporary Tensors:
  Node 434:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 28
      Input 1: 717
      Input 2: 339
    Output Tensors:
      Output 0: 718
    Intermediate Tensors:
    Temporary Tensors:
  Node 435:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 36
      Input 1: 695
      Input 2: 339
    Output Tensors:
      Output 0: 719
    Intermediate Tensors:
    Temporary Tensors:
  Node 1565:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 161
      Input 1: 274
      Input 2: 303
      Input 3: 683
      Input 4: 706
      Input 5: 718
      Input 6: 719
    Output Tensors:
      Output 0: 723
      Output 1: 724
    Intermediate Tensors:
    Temporary Tensors:
  Node 441:
    Operator: SUM
    Input Tensors:
      Input 0: 724
      Input 1: 206
    Output Tensors:
      Output 0: 725
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2112
      Temporary 1: 2113
      Temporary 2: 2114
      Temporary 3: 2115
  Node 1566:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 158
      Input 1: 159
      Input 2: 160
      Input 3: 205
      Input 4: 248
      Input 5: 281
      Input 6: 723
      Input 7: 725
    Output Tensors:
      Output 0: 737
      Output 1: 738
    Intermediate Tensors:
    Temporary Tensors:
  Node 455:
    Operator: SUM
    Input Tensors:
      Input 0: 738
      Input 1: 206
    Output Tensors:
      Output 0: 739
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2134
      Temporary 1: 2135
      Temporary 2: 2136
      Temporary 3: 2137
  Node 1567:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 157
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 247
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 737
      Input 19: 739
    Output Tensors:
      Output 0: 749
      Output 1: 760
      Output 2: 771
    Intermediate Tensors:
    Temporary Tensors:
  Node 488:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 4
      Input 1: 771
      Input 2: 339
    Output Tensors:
      Output 0: 772
    Intermediate Tensors:
    Temporary Tensors:
  Node 489:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 49
      Input 1: 749
      Input 2: 339
    Output Tensors:
      Output 0: 773
    Intermediate Tensors:
    Temporary Tensors:
  Node 1568:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 156
      Input 1: 274
      Input 2: 303
      Input 3: 737
      Input 4: 760
      Input 5: 772
      Input 6: 773
    Output Tensors:
      Output 0: 777
      Output 1: 778
    Intermediate Tensors:
    Temporary Tensors:
  Node 495:
    Operator: SUM
    Input Tensors:
      Input 0: 778
      Input 1: 206
    Output Tensors:
      Output 0: 779
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2150
      Temporary 1: 2151
      Temporary 2: 2152
      Temporary 3: 2153
  Node 1569:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 153
      Input 1: 154
      Input 2: 155
      Input 3: 205
      Input 4: 246
      Input 5: 281
      Input 6: 777
      Input 7: 779
    Output Tensors:
      Output 0: 791
      Output 1: 792
    Intermediate Tensors:
    Temporary Tensors:
  Node 509:
    Operator: SUM
    Input Tensors:
      Input 0: 792
      Input 1: 206
    Output Tensors:
      Output 0: 793
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2172
      Temporary 1: 2173
      Temporary 2: 2174
      Temporary 3: 2175
  Node 1570:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 152
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 245
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 791
      Input 19: 793
    Output Tensors:
      Output 0: 803
      Output 1: 814
      Output 2: 825
    Intermediate Tensors:
    Temporary Tensors:
  Node 542:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 16
      Input 1: 825
      Input 2: 339
    Output Tensors:
      Output 0: 826
    Intermediate Tensors:
    Temporary Tensors:
  Node 543:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 46
      Input 1: 803
      Input 2: 339
    Output Tensors:
      Output 0: 827
    Intermediate Tensors:
    Temporary Tensors:
  Node 1571:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 151
      Input 1: 274
      Input 2: 303
      Input 3: 791
      Input 4: 814
      Input 5: 826
      Input 6: 827
    Output Tensors:
      Output 0: 831
      Output 1: 832
    Intermediate Tensors:
    Temporary Tensors:
  Node 549:
    Operator: SUM
    Input Tensors:
      Input 0: 832
      Input 1: 206
    Output Tensors:
      Output 0: 833
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2188
      Temporary 1: 2189
      Temporary 2: 2190
      Temporary 3: 2191
  Node 1572:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 148
      Input 1: 149
      Input 2: 150
      Input 3: 205
      Input 4: 244
      Input 5: 281
      Input 6: 831
      Input 7: 833
    Output Tensors:
      Output 0: 845
      Output 1: 846
    Intermediate Tensors:
    Temporary Tensors:
  Node 563:
    Operator: SUM
    Input Tensors:
      Input 0: 846
      Input 1: 206
    Output Tensors:
      Output 0: 847
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2210
      Temporary 1: 2211
      Temporary 2: 2212
      Temporary 3: 2213
  Node 1573:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 147
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 243
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 845
      Input 19: 847
    Output Tensors:
      Output 0: 857
      Output 1: 868
      Output 2: 879
    Intermediate Tensors:
    Temporary Tensors:
  Node 596:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 8
      Input 1: 879
      Input 2: 339
    Output Tensors:
      Output 0: 880
    Intermediate Tensors:
    Temporary Tensors:
  Node 597:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 29
      Input 1: 857
      Input 2: 339
    Output Tensors:
      Output 0: 881
    Intermediate Tensors:
    Temporary Tensors:
  Node 1574:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 146
      Input 1: 274
      Input 2: 303
      Input 3: 845
      Input 4: 868
      Input 5: 880
      Input 6: 881
    Output Tensors:
      Output 0: 885
      Output 1: 886
    Intermediate Tensors:
    Temporary Tensors:
  Node 603:
    Operator: SUM
    Input Tensors:
      Input 0: 886
      Input 1: 206
    Output Tensors:
      Output 0: 887
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2226
      Temporary 1: 2227
      Temporary 2: 2228
      Temporary 3: 2229
  Node 1575:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 143
      Input 1: 144
      Input 2: 145
      Input 3: 205
      Input 4: 242
      Input 5: 281
      Input 6: 885
      Input 7: 887
    Output Tensors:
      Output 0: 899
      Output 1: 900
    Intermediate Tensors:
    Temporary Tensors:
  Node 617:
    Operator: SUM
    Input Tensors:
      Input 0: 900
      Input 1: 206
    Output Tensors:
      Output 0: 901
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2248
      Temporary 1: 2249
      Temporary 2: 2250
      Temporary 3: 2251
  Node 1576:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 142
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 241
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 899
      Input 19: 901
    Output Tensors:
      Output 0: 911
      Output 1: 922
      Output 2: 933
    Intermediate Tensors:
    Temporary Tensors:
  Node 650:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 52
      Input 1: 933
      Input 2: 339
    Output Tensors:
      Output 0: 934
    Intermediate Tensors:
    Temporary Tensors:
  Node 651:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 22
      Input 1: 911
      Input 2: 339
    Output Tensors:
      Output 0: 935
    Intermediate Tensors:
    Temporary Tensors:
  Node 1577:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 141
      Input 1: 274
      Input 2: 303
      Input 3: 899
      Input 4: 922
      Input 5: 934
      Input 6: 935
    Output Tensors:
      Output 0: 939
      Output 1: 940
    Intermediate Tensors:
    Temporary Tensors:
  Node 657:
    Operator: SUM
    Input Tensors:
      Input 0: 940
      Input 1: 206
    Output Tensors:
      Output 0: 941
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2264
      Temporary 1: 2265
      Temporary 2: 2266
      Temporary 3: 2267
  Node 1578:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 138
      Input 1: 139
      Input 2: 140
      Input 3: 205
      Input 4: 240
      Input 5: 281
      Input 6: 939
      Input 7: 941
    Output Tensors:
      Output 0: 953
      Output 1: 954
    Intermediate Tensors:
    Temporary Tensors:
  Node 671:
    Operator: SUM
    Input Tensors:
      Input 0: 954
      Input 1: 206
    Output Tensors:
      Output 0: 955
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2286
      Temporary 1: 2287
      Temporary 2: 2288
      Temporary 3: 2289
  Node 1579:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 137
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 239
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 953
      Input 19: 955
    Output Tensors:
      Output 0: 965
      Output 1: 976
      Output 2: 987
    Intermediate Tensors:
    Temporary Tensors:
  Node 704:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 20
      Input 1: 987
      Input 2: 339
    Output Tensors:
      Output 0: 988
    Intermediate Tensors:
    Temporary Tensors:
  Node 705:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 31
      Input 1: 965
      Input 2: 339
    Output Tensors:
      Output 0: 989
    Intermediate Tensors:
    Temporary Tensors:
  Node 1580:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 136
      Input 1: 274
      Input 2: 303
      Input 3: 953
      Input 4: 976
      Input 5: 988
      Input 6: 989
    Output Tensors:
      Output 0: 993
      Output 1: 994
    Intermediate Tensors:
    Temporary Tensors:
  Node 711:
    Operator: SUM
    Input Tensors:
      Input 0: 994
      Input 1: 206
    Output Tensors:
      Output 0: 995
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2302
      Temporary 1: 2303
      Temporary 2: 2304
      Temporary 3: 2305
  Node 1581:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 133
      Input 1: 134
      Input 2: 135
      Input 3: 205
      Input 4: 238
      Input 5: 281
      Input 6: 993
      Input 7: 995
    Output Tensors:
      Output 0: 1007
      Output 1: 1008
    Intermediate Tensors:
    Temporary Tensors:
  Node 725:
    Operator: SUM
    Input Tensors:
      Input 0: 1008
      Input 1: 206
    Output Tensors:
      Output 0: 1009
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2324
      Temporary 1: 2325
      Temporary 2: 2326
      Temporary 3: 2327
  Node 1582:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 132
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 237
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1007
      Input 19: 1009
    Output Tensors:
      Output 0: 1019
      Output 1: 1030
      Output 2: 1041
    Intermediate Tensors:
    Temporary Tensors:
  Node 758:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 35
      Input 1: 1041
      Input 2: 339
    Output Tensors:
      Output 0: 1042
    Intermediate Tensors:
    Temporary Tensors:
  Node 759:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 41
      Input 1: 1019
      Input 2: 339
    Output Tensors:
      Output 0: 1043
    Intermediate Tensors:
    Temporary Tensors:
  Node 1583:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 131
      Input 1: 274
      Input 2: 303
      Input 3: 1007
      Input 4: 1030
      Input 5: 1042
      Input 6: 1043
    Output Tensors:
      Output 0: 1047
      Output 1: 1048
    Intermediate Tensors:
    Temporary Tensors:
  Node 765:
    Operator: SUM
    Input Tensors:
      Input 0: 1048
      Input 1: 206
    Output Tensors:
      Output 0: 1049
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2340
      Temporary 1: 2341
      Temporary 2: 2342
      Temporary 3: 2343
  Node 1584:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 128
      Input 1: 129
      Input 2: 130
      Input 3: 205
      Input 4: 236
      Input 5: 281
      Input 6: 1047
      Input 7: 1049
    Output Tensors:
      Output 0: 1061
      Output 1: 1062
    Intermediate Tensors:
    Temporary Tensors:
  Node 779:
    Operator: SUM
    Input Tensors:
      Input 0: 1062
      Input 1: 206
    Output Tensors:
      Output 0: 1063
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2362
      Temporary 1: 2363
      Temporary 2: 2364
      Temporary 3: 2365
  Node 1585:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 127
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 235
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1061
      Input 19: 1063
    Output Tensors:
      Output 0: 1073
      Output 1: 1084
      Output 2: 1095
    Intermediate Tensors:
    Temporary Tensors:
  Node 812:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 33
      Input 1: 1095
      Input 2: 339
    Output Tensors:
      Output 0: 1096
    Intermediate Tensors:
    Temporary Tensors:
  Node 813:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 57
      Input 1: 1073
      Input 2: 339
    Output Tensors:
      Output 0: 1097
    Intermediate Tensors:
    Temporary Tensors:
  Node 1586:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 126
      Input 1: 274
      Input 2: 303
      Input 3: 1061
      Input 4: 1084
      Input 5: 1096
      Input 6: 1097
    Output Tensors:
      Output 0: 1101
      Output 1: 1102
    Intermediate Tensors:
    Temporary Tensors:
  Node 819:
    Operator: SUM
    Input Tensors:
      Input 0: 1102
      Input 1: 206
    Output Tensors:
      Output 0: 1103
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2378
      Temporary 1: 2379
      Temporary 2: 2380
      Temporary 3: 2381
  Node 1587:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 123
      Input 1: 124
      Input 2: 125
      Input 3: 205
      Input 4: 234
      Input 5: 281
      Input 6: 1101
      Input 7: 1103
    Output Tensors:
      Output 0: 1115
      Output 1: 1116
    Intermediate Tensors:
    Temporary Tensors:
  Node 833:
    Operator: SUM
    Input Tensors:
      Input 0: 1116
      Input 1: 206
    Output Tensors:
      Output 0: 1117
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2400
      Temporary 1: 2401
      Temporary 2: 2402
      Temporary 3: 2403
  Node 1588:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 122
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 233
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1115
      Input 19: 1117
    Output Tensors:
      Output 0: 1127
      Output 1: 1138
      Output 2: 1149
    Intermediate Tensors:
    Temporary Tensors:
  Node 866:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 45
      Input 1: 1149
      Input 2: 339
    Output Tensors:
      Output 0: 1150
    Intermediate Tensors:
    Temporary Tensors:
  Node 867:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 34
      Input 1: 1127
      Input 2: 339
    Output Tensors:
      Output 0: 1151
    Intermediate Tensors:
    Temporary Tensors:
  Node 1589:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 121
      Input 1: 274
      Input 2: 303
      Input 3: 1115
      Input 4: 1138
      Input 5: 1150
      Input 6: 1151
    Output Tensors:
      Output 0: 1155
      Output 1: 1156
    Intermediate Tensors:
    Temporary Tensors:
  Node 873:
    Operator: SUM
    Input Tensors:
      Input 0: 1156
      Input 1: 206
    Output Tensors:
      Output 0: 1157
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2416
      Temporary 1: 2417
      Temporary 2: 2418
      Temporary 3: 2419
  Node 1590:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 118
      Input 1: 119
      Input 2: 120
      Input 3: 205
      Input 4: 232
      Input 5: 281
      Input 6: 1155
      Input 7: 1157
    Output Tensors:
      Output 0: 1169
      Output 1: 1170
    Intermediate Tensors:
    Temporary Tensors:
  Node 887:
    Operator: SUM
    Input Tensors:
      Input 0: 1170
      Input 1: 206
    Output Tensors:
      Output 0: 1171
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2438
      Temporary 1: 2439
      Temporary 2: 2440
      Temporary 3: 2441
  Node 1591:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 117
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 231
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1169
      Input 19: 1171
    Output Tensors:
      Output 0: 1181
      Output 1: 1192
      Output 2: 1203
    Intermediate Tensors:
    Temporary Tensors:
  Node 920:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 3
      Input 1: 1203
      Input 2: 339
    Output Tensors:
      Output 0: 1204
    Intermediate Tensors:
    Temporary Tensors:
  Node 921:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 30
      Input 1: 1181
      Input 2: 339
    Output Tensors:
      Output 0: 1205
    Intermediate Tensors:
    Temporary Tensors:
  Node 1592:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 116
      Input 1: 274
      Input 2: 303
      Input 3: 1169
      Input 4: 1192
      Input 5: 1204
      Input 6: 1205
    Output Tensors:
      Output 0: 1209
      Output 1: 1210
    Intermediate Tensors:
    Temporary Tensors:
  Node 927:
    Operator: SUM
    Input Tensors:
      Input 0: 1210
      Input 1: 206
    Output Tensors:
      Output 0: 1211
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2454
      Temporary 1: 2455
      Temporary 2: 2456
      Temporary 3: 2457
  Node 1593:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 113
      Input 1: 114
      Input 2: 115
      Input 3: 205
      Input 4: 230
      Input 5: 281
      Input 6: 1209
      Input 7: 1211
    Output Tensors:
      Output 0: 1223
      Output 1: 1224
    Intermediate Tensors:
    Temporary Tensors:
  Node 941:
    Operator: SUM
    Input Tensors:
      Input 0: 1224
      Input 1: 206
    Output Tensors:
      Output 0: 1225
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2476
      Temporary 1: 2477
      Temporary 2: 2478
      Temporary 3: 2479
  Node 1594:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 112
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 229
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1223
      Input 19: 1225
    Output Tensors:
      Output 0: 1235
      Output 1: 1246
      Output 2: 1257
    Intermediate Tensors:
    Temporary Tensors:
  Node 974:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 10
      Input 1: 1257
      Input 2: 339
    Output Tensors:
      Output 0: 1258
    Intermediate Tensors:
    Temporary Tensors:
  Node 975:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 40
      Input 1: 1235
      Input 2: 339
    Output Tensors:
      Output 0: 1259
    Intermediate Tensors:
    Temporary Tensors:
  Node 1595:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 111
      Input 1: 274
      Input 2: 303
      Input 3: 1223
      Input 4: 1246
      Input 5: 1258
      Input 6: 1259
    Output Tensors:
      Output 0: 1263
      Output 1: 1264
    Intermediate Tensors:
    Temporary Tensors:
  Node 981:
    Operator: SUM
    Input Tensors:
      Input 0: 1264
      Input 1: 206
    Output Tensors:
      Output 0: 1265
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2492
      Temporary 1: 2493
      Temporary 2: 2494
      Temporary 3: 2495
  Node 1596:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 108
      Input 1: 109
      Input 2: 110
      Input 3: 205
      Input 4: 228
      Input 5: 281
      Input 6: 1263
      Input 7: 1265
    Output Tensors:
      Output 0: 1277
      Output 1: 1278
    Intermediate Tensors:
    Temporary Tensors:
  Node 995:
    Operator: SUM
    Input Tensors:
      Input 0: 1278
      Input 1: 206
    Output Tensors:
      Output 0: 1279
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2514
      Temporary 1: 2515
      Temporary 2: 2516
      Temporary 3: 2517
  Node 1597:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 107
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 227
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1277
      Input 19: 1279
    Output Tensors:
      Output 0: 1289
      Output 1: 1300
      Output 2: 1311
    Intermediate Tensors:
    Temporary Tensors:
  Node 1028:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 25
      Input 1: 1311
      Input 2: 339
    Output Tensors:
      Output 0: 1312
    Intermediate Tensors:
    Temporary Tensors:
  Node 1029:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 21
      Input 1: 1289
      Input 2: 339
    Output Tensors:
      Output 0: 1313
    Intermediate Tensors:
    Temporary Tensors:
  Node 1598:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 106
      Input 1: 274
      Input 2: 303
      Input 3: 1277
      Input 4: 1300
      Input 5: 1312
      Input 6: 1313
    Output Tensors:
      Output 0: 1317
      Output 1: 1318
    Intermediate Tensors:
    Temporary Tensors:
  Node 1035:
    Operator: SUM
    Input Tensors:
      Input 0: 1318
      Input 1: 206
    Output Tensors:
      Output 0: 1319
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2530
      Temporary 1: 2531
      Temporary 2: 2532
      Temporary 3: 2533
  Node 1599:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 103
      Input 1: 104
      Input 2: 105
      Input 3: 205
      Input 4: 226
      Input 5: 281
      Input 6: 1317
      Input 7: 1319
    Output Tensors:
      Output 0: 1331
      Output 1: 1332
    Intermediate Tensors:
    Temporary Tensors:
  Node 1049:
    Operator: SUM
    Input Tensors:
      Input 0: 1332
      Input 1: 206
    Output Tensors:
      Output 0: 1333
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2552
      Temporary 1: 2553
      Temporary 2: 2554
      Temporary 3: 2555
  Node 1600:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 102
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 225
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1331
      Input 19: 1333
    Output Tensors:
      Output 0: 1343
      Output 1: 1354
      Output 2: 1365
    Intermediate Tensors:
    Temporary Tensors:
  Node 1082:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 53
      Input 1: 1365
      Input 2: 339
    Output Tensors:
      Output 0: 1366
    Intermediate Tensors:
    Temporary Tensors:
  Node 1083:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 27
      Input 1: 1343
      Input 2: 339
    Output Tensors:
      Output 0: 1367
    Intermediate Tensors:
    Temporary Tensors:
  Node 1601:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 101
      Input 1: 274
      Input 2: 303
      Input 3: 1331
      Input 4: 1354
      Input 5: 1366
      Input 6: 1367
    Output Tensors:
      Output 0: 1371
      Output 1: 1372
    Intermediate Tensors:
    Temporary Tensors:
  Node 1089:
    Operator: SUM
    Input Tensors:
      Input 0: 1372
      Input 1: 206
    Output Tensors:
      Output 0: 1373
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2568
      Temporary 1: 2569
      Temporary 2: 2570
      Temporary 3: 2571
  Node 1602:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 98
      Input 1: 99
      Input 2: 100
      Input 3: 205
      Input 4: 224
      Input 5: 281
      Input 6: 1371
      Input 7: 1373
    Output Tensors:
      Output 0: 1385
      Output 1: 1386
    Intermediate Tensors:
    Temporary Tensors:
  Node 1103:
    Operator: SUM
    Input Tensors:
      Input 0: 1386
      Input 1: 206
    Output Tensors:
      Output 0: 1387
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2590
      Temporary 1: 2591
      Temporary 2: 2592
      Temporary 3: 2593
  Node 1603:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 97
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 223
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1385
      Input 19: 1387
    Output Tensors:
      Output 0: 1397
      Output 1: 1408
      Output 2: 1419
    Intermediate Tensors:
    Temporary Tensors:
  Node 1136:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 15
      Input 1: 1419
      Input 2: 339
    Output Tensors:
      Output 0: 1420
    Intermediate Tensors:
    Temporary Tensors:
  Node 1137:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 55
      Input 1: 1397
      Input 2: 339
    Output Tensors:
      Output 0: 1421
    Intermediate Tensors:
    Temporary Tensors:
  Node 1604:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 96
      Input 1: 274
      Input 2: 303
      Input 3: 1385
      Input 4: 1408
      Input 5: 1420
      Input 6: 1421
    Output Tensors:
      Output 0: 1425
      Output 1: 1426
    Intermediate Tensors:
    Temporary Tensors:
  Node 1143:
    Operator: SUM
    Input Tensors:
      Input 0: 1426
      Input 1: 206
    Output Tensors:
      Output 0: 1427
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2606
      Temporary 1: 2607
      Temporary 2: 2608
      Temporary 3: 2609
  Node 1605:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 93
      Input 1: 94
      Input 2: 95
      Input 3: 205
      Input 4: 222
      Input 5: 281
      Input 6: 1425
      Input 7: 1427
    Output Tensors:
      Output 0: 1439
      Output 1: 1440
    Intermediate Tensors:
    Temporary Tensors:
  Node 1157:
    Operator: SUM
    Input Tensors:
      Input 0: 1440
      Input 1: 206
    Output Tensors:
      Output 0: 1441
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2628
      Temporary 1: 2629
      Temporary 2: 2630
      Temporary 3: 2631
  Node 1606:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 92
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 221
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1439
      Input 19: 1441
    Output Tensors:
      Output 0: 1451
      Output 1: 1462
      Output 2: 1473
    Intermediate Tensors:
    Temporary Tensors:
  Node 1190:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 43
      Input 1: 1473
      Input 2: 339
    Output Tensors:
      Output 0: 1474
    Intermediate Tensors:
    Temporary Tensors:
  Node 1191:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 37
      Input 1: 1451
      Input 2: 339
    Output Tensors:
      Output 0: 1475
    Intermediate Tensors:
    Temporary Tensors:
  Node 1607:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 91
      Input 1: 274
      Input 2: 303
      Input 3: 1439
      Input 4: 1462
      Input 5: 1474
      Input 6: 1475
    Output Tensors:
      Output 0: 1479
      Output 1: 1480
    Intermediate Tensors:
    Temporary Tensors:
  Node 1197:
    Operator: SUM
    Input Tensors:
      Input 0: 1480
      Input 1: 206
    Output Tensors:
      Output 0: 1481
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2644
      Temporary 1: 2645
      Temporary 2: 2646
      Temporary 3: 2647
  Node 1608:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 88
      Input 1: 89
      Input 2: 90
      Input 3: 205
      Input 4: 220
      Input 5: 281
      Input 6: 1479
      Input 7: 1481
    Output Tensors:
      Output 0: 1493
      Output 1: 1494
    Intermediate Tensors:
    Temporary Tensors:
  Node 1211:
    Operator: SUM
    Input Tensors:
      Input 0: 1494
      Input 1: 206
    Output Tensors:
      Output 0: 1495
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2666
      Temporary 1: 2667
      Temporary 2: 2668
      Temporary 3: 2669
  Node 1609:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 87
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 219
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1493
      Input 19: 1495
    Output Tensors:
      Output 0: 1505
      Output 1: 1516
      Output 2: 1527
    Intermediate Tensors:
    Temporary Tensors:
  Node 1244:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 13
      Input 1: 1527
      Input 2: 339
    Output Tensors:
      Output 0: 1528
    Intermediate Tensors:
    Temporary Tensors:
  Node 1245:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 7
      Input 1: 1505
      Input 2: 339
    Output Tensors:
      Output 0: 1529
    Intermediate Tensors:
    Temporary Tensors:
  Node 1610:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 86
      Input 1: 274
      Input 2: 303
      Input 3: 1493
      Input 4: 1516
      Input 5: 1528
      Input 6: 1529
    Output Tensors:
      Output 0: 1533
      Output 1: 1534
    Intermediate Tensors:
    Temporary Tensors:
  Node 1251:
    Operator: SUM
    Input Tensors:
      Input 0: 1534
      Input 1: 206
    Output Tensors:
      Output 0: 1535
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2682
      Temporary 1: 2683
      Temporary 2: 2684
      Temporary 3: 2685
  Node 1611:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 83
      Input 1: 84
      Input 2: 85
      Input 3: 205
      Input 4: 218
      Input 5: 281
      Input 6: 1533
      Input 7: 1535
    Output Tensors:
      Output 0: 1547
      Output 1: 1548
    Intermediate Tensors:
    Temporary Tensors:
  Node 1265:
    Operator: SUM
    Input Tensors:
      Input 0: 1548
      Input 1: 206
    Output Tensors:
      Output 0: 1549
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2704
      Temporary 1: 2705
      Temporary 2: 2706
      Temporary 3: 2707
  Node 1612:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 82
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 217
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1547
      Input 19: 1549
    Output Tensors:
      Output 0: 1559
      Output 1: 1570
      Output 2: 1581
    Intermediate Tensors:
    Temporary Tensors:
  Node 1298:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 32
      Input 1: 1581
      Input 2: 339
    Output Tensors:
      Output 0: 1582
    Intermediate Tensors:
    Temporary Tensors:
  Node 1299:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 17
      Input 1: 1559
      Input 2: 339
    Output Tensors:
      Output 0: 1583
    Intermediate Tensors:
    Temporary Tensors:
  Node 1613:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 81
      Input 1: 274
      Input 2: 303
      Input 3: 1547
      Input 4: 1570
      Input 5: 1582
      Input 6: 1583
    Output Tensors:
      Output 0: 1587
      Output 1: 1588
    Intermediate Tensors:
    Temporary Tensors:
  Node 1305:
    Operator: SUM
    Input Tensors:
      Input 0: 1588
      Input 1: 206
    Output Tensors:
      Output 0: 1589
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2720
      Temporary 1: 2721
      Temporary 2: 2722
      Temporary 3: 2723
  Node 1614:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 78
      Input 1: 79
      Input 2: 80
      Input 3: 205
      Input 4: 216
      Input 5: 281
      Input 6: 1587
      Input 7: 1589
    Output Tensors:
      Output 0: 1601
      Output 1: 1602
    Intermediate Tensors:
    Temporary Tensors:
  Node 1319:
    Operator: SUM
    Input Tensors:
      Input 0: 1602
      Input 1: 206
    Output Tensors:
      Output 0: 1603
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2742
      Temporary 1: 2743
      Temporary 2: 2744
      Temporary 3: 2745
  Node 1615:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 77
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 215
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1601
      Input 19: 1603
    Output Tensors:
      Output 0: 1613
      Output 1: 1624
      Output 2: 1635
    Intermediate Tensors:
    Temporary Tensors:
  Node 1352:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 9
      Input 1: 1635
      Input 2: 339
    Output Tensors:
      Output 0: 1636
    Intermediate Tensors:
    Temporary Tensors:
  Node 1353:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 44
      Input 1: 1613
      Input 2: 339
    Output Tensors:
      Output 0: 1637
    Intermediate Tensors:
    Temporary Tensors:
  Node 1616:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 76
      Input 1: 274
      Input 2: 303
      Input 3: 1601
      Input 4: 1624
      Input 5: 1636
      Input 6: 1637
    Output Tensors:
      Output 0: 1641
      Output 1: 1642
    Intermediate Tensors:
    Temporary Tensors:
  Node 1359:
    Operator: SUM
    Input Tensors:
      Input 0: 1642
      Input 1: 206
    Output Tensors:
      Output 0: 1643
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2758
      Temporary 1: 2759
      Temporary 2: 2760
      Temporary 3: 2761
  Node 1617:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 73
      Input 1: 74
      Input 2: 75
      Input 3: 205
      Input 4: 214
      Input 5: 281
      Input 6: 1641
      Input 7: 1643
    Output Tensors:
      Output 0: 1655
      Output 1: 1656
    Intermediate Tensors:
    Temporary Tensors:
  Node 1373:
    Operator: SUM
    Input Tensors:
      Input 0: 1656
      Input 1: 206
    Output Tensors:
      Output 0: 1657
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2780
      Temporary 1: 2781
      Temporary 2: 2782
      Temporary 3: 2783
  Node 1618:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 72
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 213
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1655
      Input 19: 1657
    Output Tensors:
      Output 0: 1667
      Output 1: 1678
      Output 2: 1689
    Intermediate Tensors:
    Temporary Tensors:
  Node 1406:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 50
      Input 1: 1689
      Input 2: 339
    Output Tensors:
      Output 0: 1690
    Intermediate Tensors:
    Temporary Tensors:
  Node 1407:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 6
      Input 1: 1667
      Input 2: 339
    Output Tensors:
      Output 0: 1691
    Intermediate Tensors:
    Temporary Tensors:
  Node 1619:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 71
      Input 1: 274
      Input 2: 303
      Input 3: 1655
      Input 4: 1678
      Input 5: 1690
      Input 6: 1691
    Output Tensors:
      Output 0: 1695
      Output 1: 1696
    Intermediate Tensors:
    Temporary Tensors:
  Node 1413:
    Operator: SUM
    Input Tensors:
      Input 0: 1696
      Input 1: 206
    Output Tensors:
      Output 0: 1697
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2796
      Temporary 1: 2797
      Temporary 2: 2798
      Temporary 3: 2799
  Node 1620:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 68
      Input 1: 69
      Input 2: 70
      Input 3: 205
      Input 4: 212
      Input 5: 281
      Input 6: 1695
      Input 7: 1697
    Output Tensors:
      Output 0: 1709
      Output 1: 1710
    Intermediate Tensors:
    Temporary Tensors:
  Node 1427:
    Operator: SUM
    Input Tensors:
      Input 0: 1710
      Input 1: 206
    Output Tensors:
      Output 0: 1711
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2818
      Temporary 1: 2819
      Temporary 2: 2820
      Temporary 3: 2821
  Node 1621:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 67
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 211
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1709
      Input 19: 1711
    Output Tensors:
      Output 0: 1721
      Output 1: 1732
      Output 2: 1743
    Intermediate Tensors:
    Temporary Tensors:
  Node 1460:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 26
      Input 1: 1743
      Input 2: 339
    Output Tensors:
      Output 0: 1744
    Intermediate Tensors:
    Temporary Tensors:
  Node 1461:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 0
      Input 1: 1721
      Input 2: 339
    Output Tensors:
      Output 0: 1745
    Intermediate Tensors:
    Temporary Tensors:
  Node 1622:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 66
      Input 1: 274
      Input 2: 303
      Input 3: 1709
      Input 4: 1732
      Input 5: 1744
      Input 6: 1745
    Output Tensors:
      Output 0: 1749
      Output 1: 1750
    Intermediate Tensors:
    Temporary Tensors:
  Node 1467:
    Operator: SUM
    Input Tensors:
      Input 0: 1750
      Input 1: 206
    Output Tensors:
      Output 0: 1751
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2834
      Temporary 1: 2835
      Temporary 2: 2836
      Temporary 3: 2837
  Node 1623:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 63
      Input 1: 64
      Input 2: 65
      Input 3: 205
      Input 4: 210
      Input 5: 281
      Input 6: 1749
      Input 7: 1751
    Output Tensors:
      Output 0: 1763
      Output 1: 1764
    Intermediate Tensors:
    Temporary Tensors:
  Node 1481:
    Operator: SUM
    Input Tensors:
      Input 0: 1764
      Input 1: 206
    Output Tensors:
      Output 0: 1765
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2856
      Temporary 1: 2857
      Temporary 2: 2858
      Temporary 3: 2859
  Node 1624:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 62
      Input 1: 198
      Input 2: 199
      Input 3: 200
      Input 4: 201
      Input 5: 202
      Input 6: 203
      Input 7: 204
      Input 8: 205
      Input 9: 209
      Input 10: 266
      Input 11: 267
      Input 12: 268
      Input 13: 269
      Input 14: 270
      Input 15: 281
      Input 16: 290
      Input 17: 291
      Input 18: 1763
      Input 19: 1765
    Output Tensors:
      Output 0: 1775
      Output 1: 1786
      Output 2: 1797
    Intermediate Tensors:
    Temporary Tensors:
  Node 1514:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 56
      Input 1: 1797
      Input 2: 339
    Output Tensors:
      Output 0: 1798
    Intermediate Tensors:
    Temporary Tensors:
  Node 1515:
    Operator: DYNAMIC_UPDATE_SLICE
    Input Tensors:
      Input 0: 12
      Input 1: 1775
      Input 2: 339
    Output Tensors:
      Output 0: 1799
    Intermediate Tensors:
    Temporary Tensors:
  Node 1625:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 61
      Input 1: 274
      Input 2: 303
      Input 3: 1763
      Input 4: 1786
      Input 5: 1798
      Input 6: 1799
    Output Tensors:
      Output 0: 1803
      Output 1: 1804
    Intermediate Tensors:
    Temporary Tensors:
  Node 1521:
    Operator: SUM
    Input Tensors:
      Input 0: 1804
      Input 1: 206
    Output Tensors:
      Output 0: 1805
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2872
      Temporary 1: 2873
      Temporary 2: 2874
      Temporary 3: 2875
  Node 1626:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 58
      Input 1: 59
      Input 2: 60
      Input 3: 205
      Input 4: 208
      Input 5: 281
      Input 6: 1803
      Input 7: 1805
    Output Tensors:
      Output 0: 1817
      Output 1: 1818
    Intermediate Tensors:
    Temporary Tensors:
  Node 1535:
    Operator: SUM
    Input Tensors:
      Input 0: 1818
      Input 1: 206
    Output Tensors:
      Output 0: 1819
    Intermediate Tensors:
    Temporary Tensors:
      Temporary 0: 2894
      Temporary 1: 2895
      Temporary 2: 2896
      Temporary 3: 2897
  Node 1627:
    Operator: DELEGATE
    Input Tensors:
      Input 0: 205
      Input 1: 207
      Input 2: 281
      Input 3: 283
      Input 4: 1817
      Input 5: 1819
    Output Tensors:
      Output 0: 1825
    Intermediate Tensors:
    Temporary Tensors:'''

# Create the graph
G, node_info = parse_node_details(node_details)

# Visualize the graph
plt.figure(figsize=(25,20))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=False, node_color='lightblue', 
        node_size=3000, arrows=True, edge_color='gray')

# Add node labels
node_labels = {node: f"{node}\n{node_info[node]['operator']}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

plt.title("TensorFlow Lite Computational Graph", fontsize=16)
plt.axis('off')  # Turn off axis
plt.tight_layout(pad=0)
plt.savefig('output.png')
plt.show()  # This will block and wait for the figure to be closed
The comparative experimental results with HEAT and RoomFormer on the SceneCAD dataset are presented in the table below. The results for HEAT and RoomFormer are taken from the RoomFormer paper.

The bold numbers in the table represent the best results. As shown, our method performs the best in terms of room IoU, corner, and angle recall metrics, with precision comparable to HEAT and F1 scores comparable to RoomFormer. Although our method relies on the relationships between adjacent rooms, our results on the SceneCAD dataset, where each image contains only a single room, remain competitive.

<table border="1">
    <tr>
        <th rowspan="2" align="center" style="text-align: center; vertical-align: middle;">Method</th>
        <th rowspan="2" align="center" style="text-align: center; vertical-align: middle;">Time<br>t(s)</th>
        <th colspan="1" align="center" style="text-align: center; vertical-align: middle;">Room</th>
        <th colspan="3" align="center" style="text-align: center; vertical-align: middle;">Corner</th>
        <th colspan="3" align="center" style="text-align: center; vertical-align: middle;">Angel</th>
    </tr>
    <tr>
        <th align="center" style="text-align: center; vertical-align: middle;">IoU</th>
        <th align="center" style="text-align: center; vertical-align: middle;">Rec.</th>
        <th align="center" style="text-align: center; vertical-align: middle;">Prec.</th>
        <th align="center" style="text-align: center; vertical-align: middle;">F1</th>
        <th align="center" style="text-align: center; vertical-align: middle;">Rec.</th>
        <th align="center" style="text-align: center; vertical-align: middle;">Prec.</th>
        <th align="center" style="text-align: center; vertical-align: middle;">F1</th>
    </tr>
    <tr>
        <td align="center">HEAT</td>
        <td align="center"><u>0.12</u></td>
        <td align="center">84.9</td>
        <td align="center">79.1</td>
        <td align="center">87.8</td>
        <td align="center">83.2</td>
        <td align="center">67.8</td>
        <td align="center"><u>73.2</u></td>
        <td align="center">70.4</td>
    </tr>
    <tr>
        <td align="center">RoomFormer</td>
        <td align="center"><strong>0.01</strong></td>
        <td align="center"><u>91.7</u></td>
        <td align="center"><u>85.3</u></td>
        <td align="center"><strong>92.5</strong></td>
        <td align="center"><strong>88.8</strong></td>
        <td align="center"><u>73.7</u></td>
        <td align="center"><strong>78</strong></td>
        <td align="center"><strong>75.8</strong></td>
    </tr>
    <tr>
        <td align="center">Ours</td>
        <td align="center">0.024(0.004 + 0.02)</td>
        <td align="center"><strong>93.3</strong></td>
        <td align="center"><strong>88.4</strong></td>
        <td align="center"><u>88.3</u></td>
        <td align="center"><u>88.3</u></td>
        <td align="center"><strong>73.9</strong></td>
        <td align="center">73.1</td>
        <td align="center"><u>73.5</u></td>
    </tr>
</table>
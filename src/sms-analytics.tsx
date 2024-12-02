import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card.tsx';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { BarChart, Bar } from 'recharts';

// 샘플 데이터
const sampleData = [
  { 
    date: '월', sent: 15, received: 12, total: 27, 
    positive: 18, negative: 6, neutral: 3,
    handover: 2, manual: 1
  },
  { 
    date: '화', sent: 8, received: 10, total: 18, 
    positive: 10, negative: 5, neutral: 3,
    handover: 1, manual: 1
  },
  { 
    date: '수', sent: 12, received: 15, total: 27, 
    positive: 15, negative: 8, neutral: 4,
    handover: 2, manual: 1
  },
  { 
    date: '목', sent: 10, received: 8, total: 18, 
    positive: 8, negative: 7, neutral: 3,
    handover: 1, manual: 1
  },
  { 
    date: '금', sent: 20, received: 18, total: 38, 
    positive: 22, negative: 10, neutral: 6,
    handover: 3, manual: 2
  },
  { 
    date: '토', sent: 5, received: 7, total: 12, 
    positive: 7, negative: 3, neutral: 2,
    handover: 1, manual: 0
  },
  { 
    date: '일', sent: 3, received: 4, total: 7, 
    positive: 4, negative: 2, neutral: 1,
    handover: 0, manual: 1
  }
];

const SENTIMENT_COLORS = {
  positive: '#22c55e',  // 초록색
  negative: '#ef4444',  // 빨간색
  neutral: '#94a3b8'    // 회색
};

const RESPONSE_COLORS = {
  auto: '#3b82f6',      // 파란색
  handover: '#f97316',  // 주황색
  manual: '#8b5cf6'     // 보라색
};

const SMSAnalytics = () => {
  const [data] = useState(sampleData);
  
  const totalSent = data.reduce((acc, curr) => acc + curr.sent, 0);
  const totalReceived = data.reduce((acc, curr) => acc + curr.received, 0);
  const totalMessages = totalSent + totalReceived;
  const totalHandover = data.reduce((acc, curr) => acc + curr.handover, 0);
  const totalManual = data.reduce((acc, curr) => acc + curr.manual, 0);
  const totalAuto = totalMessages - totalHandover - totalManual;
  
  // 전체 감성 분석 데이터 계산
  const totalSentiment = data.reduce((acc, curr) => ({
    positive: acc.positive + curr.positive,
    negative: acc.negative + curr.negative,
    neutral: acc.neutral + curr.neutral
  }), { positive: 0, negative: 0, neutral: 0 });

  const sentimentPieData = [
    { name: '긍정적', value: totalSentiment.positive },
    { name: '부정적', value: totalSentiment.negative },
    { name: '중립적', value: totalSentiment.neutral }
  ];

  const responsePieData = [
    { name: '자동응답', value: totalAuto },
    { name: '핸드오버', value: totalHandover },
    { name: '수동응답', value: totalManual }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-4">
      <h1 className="text-2xl font-bold mb-4">SMS 주간 통계</h1>
      
      {/* 요약 카드들 */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-lg font-semibold">총 발신</div>
            <div className="text-3xl font-bold text-blue-600">{totalSent}건</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-lg font-semibold">총 수신</div>
            <div className="text-3xl font-bold text-green-600">{totalReceived}건</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-lg font-semibold">핸드오버</div>
            <div className="text-3xl font-bold text-orange-600">{totalHandover}건</div>
            <div className="text-sm text-gray-500">
              ({((totalHandover/totalMessages) * 100).toFixed(1)}%)
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-lg font-semibold">수동응답</div>
            <div className="text-3xl font-bold text-purple-600">{totalManual}건</div>
            <div className="text-sm text-gray-500">
              ({((totalManual/totalMessages) * 100).toFixed(1)}%)
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-lg font-semibold">총 메시지</div>
            <div className="text-3xl font-bold text-gray-600">{totalMessages}건</div>
          </CardContent>
        </Card>
      </div>

      {/* 라인 차트 */}
      <Card>
        <CardHeader>
          <CardTitle>일별 메시지 추이</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="sent" stroke="#2563eb" name="발신" />
                <Line type="monotone" dataKey="received" stroke="#16a34a" name="수신" />
                <Line type="monotone" dataKey="handover" stroke="#f97316" name="핸드오버" />
                <Line type="monotone" dataKey="manual" stroke="#8b5cf6" name="수동응답" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* 감성 분석과 응답 유형 섹션 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 감성 분석 파이 차트 */}
        <Card>
          <CardHeader>
            <CardTitle>메시지 감성 분석</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    labelLine={true}
                    label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    <Cell fill={SENTIMENT_COLORS.positive} />
                    <Cell fill={SENTIMENT_COLORS.negative} />
                    <Cell fill={SENTIMENT_COLORS.neutral} />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* 응답 유형 파이 차트 */}
        <Card>
          <CardHeader>
            <CardTitle>응답 유형 분석</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={responsePieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    labelLine={true}
                    label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    <Cell fill={RESPONSE_COLORS.auto} />
                    <Cell fill={RESPONSE_COLORS.handover} />
                    <Cell fill={RESPONSE_COLORS.manual} />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* 일별 감성 추이 */}
        <Card>
          <CardHeader>
            <CardTitle>일별 메시지 감성 추이</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="positive" stroke={SENTIMENT_COLORS.positive} name="긍정적" />
                  <Line type="monotone" dataKey="negative" stroke={SENTIMENT_COLORS.negative} name="부정적" />
                  <Line type="monotone" dataKey="neutral" stroke={SENTIMENT_COLORS.neutral} name="중립적" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* 응답 유형 일별 추이 */}
        <Card>
          <CardHeader>
            <CardTitle>일별 응답 유형 추이</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="handover" stackId="a" fill={RESPONSE_COLORS.handover} name="핸드오버" />
                  <Bar dataKey="manual" stackId="a" fill={RESPONSE_COLORS.manual} name="수동응답" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default SMSAnalytics;

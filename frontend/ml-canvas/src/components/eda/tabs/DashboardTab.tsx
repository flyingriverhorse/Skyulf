import React from 'react';
import { OverviewCards } from '../OverviewCards';
import { AlertsSection } from '../AlertsSection';

interface DashboardTabProps {
    profile: any;
}

export const DashboardTab: React.FC<DashboardTabProps> = ({ profile }) => {
    return (
        <div className="space-y-6">
            <OverviewCards profile={profile} />
            <AlertsSection alerts={profile.alerts} />
        </div>
    );
};

import { Info, Shield } from 'lucide-react';
import type { EDAAlert } from '../../../core/types/edaProfile';

interface PIIReviewTabProps {
  alerts?: EDAAlert[];
}

/** Review detector metadata without rendering alert messages or dataset values. */
export function PIIReviewTab({ alerts = [] }: PIIReviewTabProps) {
  const findings = alerts.filter(alert => alert.type === 'PII');
  const columnCount = new Set(findings.map(alert => alert.column).filter(Boolean)).size;

  return (
    <section id="eda-pii-review" aria-labelledby="eda-pii-heading" className="space-y-6">
      <div>
        <h2 id="eda-pii-heading" className="flex items-center gap-2 text-xl font-semibold text-gray-900 dark:text-white">
          <Shield aria-hidden="true" className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          PII review
        </h2>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
          Review columns that may contain personally identifiable information (PII).
          Values and samples are not shown here.
        </p>
      </div>

      <div className="flex gap-3 rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
        <Info aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-blue-700 dark:text-blue-300" />
        <div className="space-y-1 text-sm text-blue-900 dark:text-blue-100">
          <p className="font-semibold">Advisory heuristics only</p>
          <p>
            Matches can be false positives, and detection can miss personal information.
            Findings do not classify data as legally sensitive or determine compliance.
            This review does not mask, delete, or block data.
          </p>
        </div>
      </div>

      {findings.length === 0 ? (
        <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
          <h3 className="font-semibold text-gray-900 dark:text-white">No PII findings recorded</h3>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
            This profile contains no PII alerts. That does not guarantee the dataset is free of personal information.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
          <div className="border-b border-gray-200 p-4 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {findings.length} finding{findings.length === 1 ? '' : 's'} across {columnCount} flagged column{columnCount === 1 ? '' : 's'}
            </h3>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
              The detector checks email and phone patterns together; it does not report which pattern matched.
              Findings describe the saved profile, including columns with pending exclusions.
            </p>
          </div>
          <div
            role="region"
            aria-label="PII findings table"
            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Keyboard users need to scroll the table horizontally on narrow screens.
            tabIndex={0}
            className="overflow-x-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500"
          >
            <table className="w-full min-w-[600px] text-left text-sm">
              <caption className="sr-only">Possible PII findings from this profile</caption>
              <thead className="bg-gray-50 text-gray-600 dark:bg-gray-900/50 dark:text-gray-300">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Column</th>
                  <th scope="col" className="px-4 py-3 font-medium">Detector category</th>
                  <th scope="col" className="px-4 py-3 font-medium">Severity</th>
                  <th scope="col" className="px-4 py-3 font-medium">Explanation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {findings.map((alert, index) => {
                  // Only known severity labels may reach the UI; messages can contain raw values.
                  const severity = alert.severity === 'error' ? 'Error'
                    : alert.severity === 'warning' ? 'Warning'
                      : alert.severity === 'info' ? 'Info' : 'Unspecified';
                  const severityClass = severity === 'Error'
                    ? 'bg-red-50 text-red-800 dark:bg-red-900/30 dark:text-red-200'
                    : severity === 'Warning'
                      ? 'bg-amber-50 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200'
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200';

                  return (
                    <tr key={index} className="align-top">
                      <th scope="row" className="max-w-64 break-words px-4 py-4 font-medium text-gray-900 dark:text-white">
                        {alert.column || 'Column not specified'}
                      </th>
                      <td className="whitespace-nowrap px-4 py-4 text-gray-700 dark:text-gray-200">Email / phone</td>
                      <td className="px-4 py-4">
                        <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${severityClass}`}>{severity}</span>
                      </td>
                      <td className="px-4 py-4 text-gray-600 dark:text-gray-300">
                        Values may match an email address or phone number pattern. Review whether this column is appropriate for your intended use.
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
